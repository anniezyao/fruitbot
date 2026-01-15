from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from utils import (
    cancel_order,
    get_book,
    get_open_orders,
    get_position,
    get_status,
    get_tas,
    get_tick,
    send_order,
)


TICK = 0.01
POSTERIOR = 500
MIN_TOTAL_QTY = 5.0
EPS = 1e-9
K = 0.05
FIT_MIN = 0.10
FIT_MAX = 0.85
FIT_EMA_LAMBDA = 0.10
FIT_VOL_SAT = 50.0
FIT_NOISE_K = 0.05


def round_tick(p: float) -> float:
    return math.floor(p / TICK) * TICK


def calc_fair(fit):
    fit = max(0.0, min(1.0, float(fit)))

    tas = get_tas('APPL', 5) or []
    if not tas:
        fair = POSTERIOR
        fit = (1.0 - FIT_EMA_LAMBDA) * fit + FIT_EMA_LAMBDA * FIT_MIN
        return fit, fair

    prices, qtys = [], []
    for t in tas:
        p = t.get("price")
        q = t.get("quantity")
        if p is None or q is None or q <= 0:
            continue
        prices.append(float(p))
        qtys.append(float(q))

    if not prices:
        fair = POSTERIOR
        fit = (1.0 - FIT_EMA_LAMBDA) * fit + FIT_EMA_LAMBDA * FIT_MIN
        return fit, fair

    total_qty = sum(qtys)
    if total_qty < MIN_TOTAL_QTY:
        fair = POSTERIOR
        fit = (1.0 - FIT_EMA_LAMBDA) * fit + FIT_EMA_LAMBDA * FIT_MIN
        return fit, fair

    vwap = sum(p * q for p, q in zip(prices, qtys)) / total_qty

    prices_sorted = sorted(prices)
    n = len(prices_sorted)
    if n % 2 == 1:
        median = prices_sorted[n // 2]
    else:
        median = 0.5 * (prices_sorted[n // 2 - 1] + prices_sorted[n // 2])

    abs_devs = sorted(abs(p - median) for p in prices)
    if n % 2 == 1:
        mad = abs_devs[n // 2]
    else:
        mad = 0.5 * (abs_devs[n // 2 - 1] + abs_devs[n // 2])

    sigma = 1.4826 * mad
    shrink = (K * K) / (K * K + sigma * sigma + EPS)
    market_signal = shrink * vwap + (1.0 - shrink) * median

    vol_score = total_qty / (total_qty + FIT_VOL_SAT)
    noise_score = (FIT_NOISE_K * FIT_NOISE_K) / (FIT_NOISE_K * FIT_NOISE_K + sigma * sigma + EPS)
    c = vol_score * noise_score
    fit_inst = FIT_MIN + c * (FIT_MAX - FIT_MIN)
    fit = (1.0 - FIT_EMA_LAMBDA) * fit + FIT_EMA_LAMBDA * fit_inst
    fit = max(FIT_MIN, min(FIT_MAX, fit))

    fair = (1 - fit) * POSTERIOR + fit * market_signal
    print(fit, fair)
    return fit, fair


@dataclass
class BotConfig:
    ticker: str = 'APPL'
    loop_sleep_s: float = 0.25
    base_size: int = 10
    min_size: int = 1
    max_pos: int = 100
    min_spread_ticks: int = 2
    edge_ticks: int = 1
    skew_ticks: int = 6
    cancel_threshold_ticks: int = 1
    requote_cooldown_ms: int = 400
    max_orders_per_side: int = 1


@dataclass
class OrderRef:
    order_id: int
    side: str
    price: float
    qty: int
    status: str = 'OPEN'
    ts_ms: int = 0


@dataclass
class Snapshot:
    tick: int
    position: int
    book: Dict[str, Any]
    open_orders: List[Dict[str, Any]]


@dataclass
class Quote:
    price: float
    qty: int


@dataclass
class QuotePlan:
    desired_bid: Optional[Quote]
    desired_ask: Optional[Quote]
    cancel_ids: List[int] = field(default_factory=list)
    place: List[Tuple[str, Quote]] = field(default_factory=list)
    replace: List[Tuple[int, str, Quote]] = field(default_factory=list)


@dataclass
class BotState:
    fit: float = FIT_MIN
    last_action_ms: Dict[str, int] = field(default_factory=lambda: {'BUY': 0, 'SELL': 0})
    last_intent: Dict[str, Optional[Quote]] = field(default_factory=lambda: {'BUY': None, 'SELL': None})


def _now_ms() -> int:
    return int(time.time() * 1000)


def _best_prices(book: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    bids = book.get('bids') or []
    asks = book.get('asks') or book.get('ask') or []
    best_bid = None
    best_ask = None
    if bids:
        best_bid = float(bids[0].get('price', bids[0].get('px', bids[0]))) if isinstance(bids[0], dict) else float(bids[0])
    if asks:
        best_ask = float(asks[0].get('price', asks[0].get('px', asks[0]))) if isinstance(asks[0], dict) else float(asks[0])
    return best_bid, best_ask


def observe(cfg: BotConfig) -> Snapshot:
    tick = get_tick()
    position = int(get_position(cfg.ticker) or 0)
    book = get_book(cfg.ticker)
    open_orders = get_open_orders(cfg.ticker) or []
    return Snapshot(tick=tick, position=position, book=book, open_orders=open_orders)


def compute_inventory_skew(position: int, cfg: BotConfig) -> float:
    if cfg.max_pos <= 0:
        return 0.0
    x = position / float(cfg.max_pos)
    return max(-1.0, min(1.0, x))


def compute_base_halfspread_ticks(best_bid: Optional[float], best_ask: Optional[float], cfg: BotConfig) -> int:
    if best_bid is None or best_ask is None:
        return max(cfg.min_spread_ticks // 2, 1)
    spread = best_ask - best_bid
    spread_ticks = int(round(spread / TICK))
    half = max(cfg.min_spread_ticks // 2, max(1, spread_ticks // 2))
    return max(half, 1)


def compute_desired_quotes(fair: float, snap: Snapshot, cfg: BotConfig) -> Tuple[Optional[Quote], Optional[Quote]]:
    best_bid, best_ask = _best_prices(snap.book)
    half = compute_base_halfspread_ticks(best_bid, best_ask, cfg)
    raw_bid = fair - (half + cfg.edge_ticks) * TICK
    raw_ask = fair + (half + cfg.edge_ticks) * TICK

    skew = compute_inventory_skew(snap.position, cfg)
    bid_px = raw_bid - skew * cfg.skew_ticks * TICK
    ask_px = raw_ask - skew * cfg.skew_ticks * TICK

    bid_px = round_tick(bid_px)
    ask_px = round_tick(ask_px)

    if best_ask is not None:
        bid_px = min(bid_px, round_tick(best_ask - TICK))
    if best_bid is not None:
        ask_px = max(ask_px, round_tick(best_bid + TICK))

    util = compute_inventory_skew(snap.position, cfg)
    bid_size = int(round(cfg.base_size * max(0.0, min(1.0, 1.0 - util))))
    ask_size = int(round(cfg.base_size * max(0.0, min(1.0, 1.0 + util))))
    bid_size = max(cfg.min_size, bid_size)
    ask_size = max(cfg.min_size, ask_size)

    desired_bid = Quote(price=bid_px, qty=bid_size)
    desired_ask = Quote(price=ask_px, qty=ask_size)

    if snap.position >= cfg.max_pos:
        desired_bid = None
    if snap.position <= -cfg.max_pos:
        desired_ask = None

    return desired_bid, desired_ask


def _normalize_open_orders(open_orders: List[Dict[str, Any]]) -> Dict[str, List[OrderRef]]:
    out: Dict[str, List[OrderRef]] = {'BUY': [], 'SELL': []}
    for o in open_orders or []:
        side = o.get('action') or o.get('side')
        side = str(side).upper()
        if side not in out:
            continue
        oid = o.get('order_id') or o.get('id') or o.get('orderId')
        if oid is None:
            continue
        price = float(o.get('price', 0.0))
        qty = int(o.get('quantity', o.get('qty', 0)) or 0)
        status = o.get('status', 'OPEN')
        out[side].append(OrderRef(order_id=int(oid), side=side, price=price, qty=qty, status=status))
    out['BUY'].sort(key=lambda x: (-x.price, x.order_id))
    out['SELL'].sort(key=lambda x: (x.price, x.order_id))
    return out


def _ticks_diff(a: float, b: float) -> int:
    return int(round(abs(a - b) / TICK))


def make_plan(state: BotState, fair: float, snap: Snapshot, cfg: BotConfig) -> QuotePlan:
    desired_bid, desired_ask = compute_desired_quotes(fair, snap, cfg)
    by_side = _normalize_open_orders(snap.open_orders)
    now_ms = _now_ms()
    plan = QuotePlan(desired_bid=desired_bid, desired_ask=desired_ask)

    def handle_side(side: str, desired: Optional[Quote]):
        existing = by_side.get(side, [])[: cfg.max_orders_per_side]
        extra = by_side.get(side, [])[cfg.max_orders_per_side :]
        for o in extra:
            plan.cancel_ids.append(o.order_id)

        if desired is None:
            for o in existing:
                plan.cancel_ids.append(o.order_id)
            state.last_intent[side] = None
            return

        if not existing:
            if now_ms - state.last_action_ms[side] >= cfg.requote_cooldown_ms:
                plan.place.append((side, desired))
                state.last_intent[side] = desired
            return

        o = existing[0]
        need_requote = False
        if _ticks_diff(o.price, desired.price) >= cfg.cancel_threshold_ticks:
            need_requote = True
        if o.qty != desired.qty:
            need_requote = True

        if need_requote and (now_ms - state.last_action_ms[side] >= cfg.requote_cooldown_ms):
            plan.replace.append((o.order_id, side, desired))
            state.last_intent[side] = desired
        else:
            state.last_intent[side] = Quote(price=o.price, qty=o.qty)

    handle_side('BUY', desired_bid)
    handle_side('SELL', desired_ask)
    return plan


def execute_plan(plan: QuotePlan, state: BotState, cfg: BotConfig) -> None:
    now_ms = _now_ms()

    for oid in plan.cancel_ids:
        try:
            cancel_order(oid)
        except Exception:
            pass

    for oid, side, q in plan.replace:
        try:
            cancel_order(oid)
        except Exception:
            pass
        try:
            send_order(cfg.ticker, side, q.price, q.qty)
            state.last_action_ms[side] = now_ms
        except Exception:
            pass

    for side, q in plan.place:
        try:
            send_order(cfg.ticker, side, q.price, q.qty)
            state.last_action_ms[side] = now_ms
        except Exception:
            pass


def step(state: BotState, cfg: BotConfig) -> Tuple[BotState, Optional[QuotePlan], Optional[float]]:
    if get_status() not in (None, 'ACTIVE', 'RUNNING'):
        return state, None, None
    snap = observe(cfg)
    state.fit, fair = calc_fair(state.fit)
    plan = make_plan(state, fair, snap, cfg)
    execute_plan(plan, state, cfg)
    return state, plan, fair


def run_forever(cfg: Optional[BotConfig] = None, state: Optional[BotState] = None) -> None:
    cfg = cfg or BotConfig()
    state = state or BotState()
    while True:
        try:
            state, _, _ = step(state, cfg)
        except Exception:
            pass
        time.sleep(cfg.loop_sleep_s)


def run_n_steps(n: int, cfg: Optional[BotConfig] = None, state: Optional[BotState] = None) -> BotState:
    cfg = cfg or BotConfig()
    state = state or BotState()
    for _ in range(int(n)):
        state, _, _ = step(state, cfg)
        time.sleep(cfg.loop_sleep_s)
    return state

run_forever()

