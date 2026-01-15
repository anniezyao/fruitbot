from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from config import *

from utils import (
    cancel_order,
    get_book,
    get_open_orders,
    get_position,
    get_status,
    get_tick,
    send_order,
)


TICK = 0.01
POSTERIOR_DEFAULT = 500.0
FAIR_CLAMP_TICKS = 50  # Clamp fair within ±50 ticks of book-mid

THEO_PATH = "theo"


def _read_fair_from_file(path: str = THEO_PATH) -> float:
    """Read fair value from a plain-text file.

    Assumes theo is always correct; no fallback needed.
    """
    try:
        s = Path(path).read_text().strip()
        v = float(s)
        if math.isfinite(v):
            return v
    except Exception as e:
        print(f"Error reading theo file: {e}")
    return POSTERIOR_DEFAULT  # Fallback if error, but assume correct


def round_tick(p: float) -> float:
    return math.floor(p / TICK) * TICK


@dataclass
class BotConfig:
    ticker: str = TICKER
    loop_sleep_s: float = 0.25
    base_size: int = 500
    min_size: int = 100
    max_pos: int = 200000
    min_spread_ticks: int = 2
    edge_ticks: int = 10  # Increased to 10 to further avoid pennying
    skew_ticks: int = 6  # Added for inventory skew like Bot2
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
    trader_id: Optional[str] = None


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
    """Observe the current market state by fetching tick, position, order book, and open orders for the ticker."""
    tick = get_tick()
    position = int(get_position(cfg.ticker) or 0)
    book = get_book(cfg.ticker)
    open_orders = get_open_orders(cfg.ticker) or []
    return Snapshot(tick=tick, position=position, book=book, open_orders=open_orders)


def _normalize_open_orders(open_orders: List[Dict[str, Any]]) -> Dict[str, List[OrderRef]]:
    """Normalize a list of open orders into a dictionary keyed by side ('BUY' or 'SELL'), 
    with each value being a sorted list of OrderRef objects. Orders are sorted by price 
    (descending for BUY, ascending for SELL) and then by order ID."""
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
        trader_id = o.get('trader_id') or o.get('traderId')
        out[side].append(OrderRef(order_id=int(oid), side=side, price=price, qty=qty, status=status, trader_id=trader_id))
    out['BUY'].sort(key=lambda x: (-x.price, x.order_id))
    out['SELL'].sort(key=lambda x: (x.price, x.order_id))
    return out


def _is_our_order(order: OrderRef) -> bool:
    """Check if the given order belongs to our trader to avoid self-trading."""
    return order.trader_id == TRADER_ID


def _ticks_diff(a: float, b: float) -> int:
    return int(round(abs(a - b) / TICK))


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
    # Clamp fair within ±FAIR_CLAMP_TICKS of book mid
    if best_bid is not None and best_ask is not None:
        mid = (best_bid + best_ask) / 2
        clamp_min = mid - FAIR_CLAMP_TICKS * TICK
        clamp_max = mid + FAIR_CLAMP_TICKS * TICK
        fair = max(clamp_min, min(clamp_max, fair))
    
    half = compute_base_halfspread_ticks(best_bid, best_ask, cfg)
    raw_bid = fair - (half + cfg.edge_ticks) * TICK
    raw_ask = fair + (half + cfg.edge_ticks) * TICK

    skew = compute_inventory_skew(snap.position, cfg)
    bid_px = raw_bid - skew * cfg.skew_ticks * TICK
    ask_px = raw_ask - skew * cfg.skew_ticks * TICK

    bid_px = round_tick(bid_px)
    ask_px = round_tick(ask_px)

    # Clamp prices to [0, 1000]
    bid_px = max(0.0, min(1000.0, bid_px))
    ask_px = max(0.0, min(1000.0, ask_px))

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


def make_plan(state: BotState, fair: float, snap: Snapshot, cfg: BotConfig) -> QuotePlan:
    desired_bid, desired_ask = compute_desired_quotes(fair, snap, cfg)
    by_side = _normalize_open_orders(snap.open_orders)
    now_ms = _now_ms()
    plan = QuotePlan(desired_bid=desired_bid, desired_ask=desired_ask)

    # Handle each side
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
        need_requote = _ticks_diff(o.price, desired.price) >= cfg.cancel_threshold_ticks or o.qty != desired.qty

        if need_requote and (now_ms - state.last_action_ms[side] >= cfg.requote_cooldown_ms):
            plan.replace.append((o.order_id, side, desired))
            state.last_intent[side] = desired
        else:
            state.last_intent[side] = Quote(price=o.price, qty=o.qty)

    handle_side('BUY', desired_bid)
    handle_side('SELL', desired_ask)
    return plan


def execute_plan(plan: QuotePlan, state: BotState, cfg: BotConfig) -> None:
    """Execute the given quote plan by canceling specified orders, replacing existing ones, and placing new ones.
    
    This function attempts to cancel orders first, then for replacements, cancels the old order and places a new one,
    and finally places new orders. It updates the bot state's last action timestamps on success and logs exceptions."""
    now_ms = _now_ms()

    for oid in plan.cancel_ids:
        try:
            cancel_order(oid)
        except Exception as e:
            print(f"Error canceling order {oid}: {e}")

    for oid, side, q in plan.replace:
        try:
            cancel_order(oid)
        except Exception as e:
            print(f"Error canceling order {oid} for replace: {e}")
        try:
            send_order(cfg.ticker, side, q.price, q.qty)
            state.last_action_ms[side] = now_ms
        except Exception as e:
            print(f"Error placing order for replace: {e}")

    for side, q in plan.place:
        try:
            send_order(cfg.ticker, side, q.price, q.qty)
            state.last_action_ms[side] = now_ms
        except Exception as e:
            print(f"Error placing order: {e}")


def step(state: BotState, cfg: BotConfig) -> Tuple[BotState, Optional[QuotePlan], Optional[float]]:
    """Perform a single step of the bot: check status, read fair value, observe market, make and execute a plan.
    
    If the trading status is not active, returns early without changes. Otherwise, fetches fair value,
    observes the market snapshot, generates a quote plan, executes it, and returns the updated state,
    the plan, and the fair value."""
    if get_status() not in (None, 'ACTIVE', 'RUNNING'):
        return state, None, None
    fair = _read_fair_from_file()
    snap = observe(cfg)
    plan = make_plan(state, fair, snap, cfg)
    execute_plan(plan, state, cfg)
    return state, plan, fair


def main(cfg: Optional[BotConfig] = None, state: Optional[BotState] = None) -> None:
    cfg = cfg or BotConfig()
    state = state or BotState()
    while True:
        try:
            state, _, _ = step(state, cfg)
        except Exception as e:
            print(f"Error in main loop: {e}")
        time.sleep(cfg.loop_sleep_s)

main()
