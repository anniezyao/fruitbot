from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config import *  # noqa: F401,F403

from utils import (
    cancel_all_orders,
    get_book,
    get_open_orders,
    get_position,
    get_status,
    get_tick,
    send_order,
)

TICK = 0.01

# If your exchange charges fees / spreads are > 1 tick, a 1-tick threshold is usually too low.
# Keep small if you want frequency, but know it increases "false arb".
THRESHOLD = 0.02
STAT_THRESHOLD = 0.25

MEAN_FAIR = 500.0
CORR = 0.82
FCALL_FAIR = 80.0


@dataclass
class ArbConfig:
    tickers: List[str] = field(default_factory=lambda: ["FRUIT", "APPL", "ORNG", "FCALL"])
    loop_sleep_s: float = 0.25
    base_size: int = 100
    max_pos: int = 200_000

    # Order management (stops the "spam/cancel every loop" problem)
    refresh_ticks: int = 3              # only refresh orders every N ticks
    min_price_move_ticks: int = 1       # only replace if desired px moved by >= this
    cancel_if_no_action_ticks: int = 10 # periodically clear stale orders if we stop trading


@dataclass
class Snapshot:
    tick: int
    positions: Dict[str, int]
    books: Dict[str, Dict[str, Any]]
    open_orders: Dict[str, List[Dict[str, Any]]]


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _extract_side_top(side: Any) -> Optional[float]:
    """
    Attempts to extract a best price from many common book formats.

    Accepts:
      - list of dicts: [{'price': 1.23, 'quantity': 10}, ...]
      - list of floats: [1.23, 1.22, ...]
      - single dict: {'price': 1.23, ...}
      - single float: 1.23
      - list of tuples: [(1.23, 10), ...]
    """
    if side is None:
        return None

    # single float / int
    if isinstance(side, (int, float)):
        return float(side)

    # single dict
    if isinstance(side, dict):
        return _to_float(side.get("price", side.get("px")))

    # list / tuple
    if isinstance(side, (list, tuple)):
        if not side:
            return None
        top = side[0]
        if isinstance(top, (int, float)):
            return float(top)
        if isinstance(top, dict):
            return _to_float(top.get("price", top.get("px")))
        if isinstance(top, (list, tuple)) and top:
            # (price, qty) style
            return _to_float(top[0])
        return None

    return None


def _best_prices(book: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    Very defensive parsing:
      - bids: book['bids'] or book['bid']
      - asks: book['asks'] or book['ask']
    """
    bids = book.get("bids", None)
    if bids is None:
        bids = book.get("bid", None)

    asks = book.get("asks", None)
    if asks is None:
        asks = book.get("ask", None)

    best_bid = _extract_side_top(bids)
    best_ask = _extract_side_top(asks)
    return best_bid, best_ask


def observe(cfg: ArbConfig) -> Snapshot:
    tick = int(get_tick())
    positions = {t: int(get_position(t) or 0) for t in cfg.tickers}
    books = {t: (get_book(t) or {}) for t in cfg.tickers}
    open_orders = {t: (get_open_orders(t) or []) for t in cfg.tickers}
    return Snapshot(tick=tick, positions=positions, books=books, open_orders=open_orders)


# ---------- Risk / sanity helpers ----------

def _within_limits(pos: int, delta: int, max_pos: int) -> bool:
    new_pos = pos + delta
    return (-max_pos) <= new_pos <= max_pos


def _have_prices(pr: Dict[str, Dict[str, Optional[float]]], ticker: str) -> bool:
    return pr[ticker]["bid"] is not None and pr[ticker]["ask"] is not None


def _mid(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is None or ask is None:
        return None
    return (bid + ask) / 2.0


# ---------- Strategy: build desired actions (prioritized) ----------

Action = Tuple[str, str, float, int]  # (ticker, side, price, qty)


def compute_actions(snap: Snapshot, cfg: ArbConfig) -> List[Action]:
    books = snap.books
    pos = snap.positions

    # Parse best bid/ask
    px: Dict[str, Dict[str, Optional[float]]] = {}
    for t in cfg.tickers:
        bid, ask = _best_prices(books[t])
        px[t] = {"bid": bid, "ask": ask}

    # Optional: basic sanity check to avoid "arbing wrong direction" from inverted books
    for t in cfg.tickers:
        b, a = px[t]["bid"], px[t]["ask"]
        if b is not None and a is not None and b > a:
            # If your exchange can show crossed books, remove this.
            print(f"[WARN] Crossed book detected for {t}: bid {b} > ask {a}. Skipping this tick.")
            return []

    actions: List[Action] = []

    # --- 1) Triangular arb: FRUIT ≈ APPL + ORNG (highest priority) ---
    fruit_bid, fruit_ask = px["FRUIT"]["bid"], px["FRUIT"]["ask"]
    appl_bid, appl_ask = px["APPL"]["bid"], px["APPL"]["ask"]
    orng_bid, orng_ask = px["ORNG"]["bid"], px["ORNG"]["ask"]

    # SELL FRUIT, BUY APPL+ORNG if FRUIT too rich vs synthetic ask
    if fruit_bid is not None and appl_ask is not None and orng_ask is not None:
        synthetic_ask = appl_ask + orng_ask
        if fruit_bid > synthetic_ask + THRESHOLD:
            qty = cfg.base_size
            if (
                _within_limits(pos["FRUIT"], -qty, cfg.max_pos)
                and _within_limits(pos["APPL"], +qty, cfg.max_pos)
                and _within_limits(pos["ORNG"], +qty, cfg.max_pos)
            ):
                actions.append(("FRUIT", "SELL", float(fruit_bid), qty))
                actions.append(("APPL", "BUY", float(appl_ask), qty))
                actions.append(("ORNG", "BUY", float(orng_ask), qty))
                return actions  # IMPORTANT: prioritize; avoid mixing with other strategies this tick

    # BUY FRUIT, SELL APPL+ORNG if FRUIT too cheap vs synthetic bid
    if fruit_ask is not None and appl_bid is not None and orng_bid is not None:
        synthetic_bid = appl_bid + orng_bid
        if fruit_ask < synthetic_bid - THRESHOLD:
            qty = cfg.base_size
            if (
                _within_limits(pos["FRUIT"], +qty, cfg.max_pos)
                and _within_limits(pos["APPL"], -qty, cfg.max_pos)
                and _within_limits(pos["ORNG"], -qty, cfg.max_pos)
            ):
                actions.append(("FRUIT", "BUY", float(fruit_ask), qty))
                actions.append(("APPL", "SELL", float(appl_bid), qty))
                actions.append(("ORNG", "SELL", float(orng_bid), qty))
                return actions  # prioritize

    # --- 2) FCALL hedge arb (second priority) ---
    fcall_bid, fcall_ask = px["FCALL"]["bid"], px["FCALL"]["ask"]

    # Only trade FCALL arb if we have FRUIT price to hedge
    if fcall_bid is not None and px["FRUIT"]["ask"] is not None:
        if fcall_bid > FCALL_FAIR + THRESHOLD:
            fqty = cfg.base_size
            hqty = 10 * fqty  # hedge with 10 FRUIT
            if (
                _within_limits(pos["FCALL"], -fqty, cfg.max_pos)
                and _within_limits(pos["FRUIT"], +hqty, cfg.max_pos)
            ):
                actions.append(("FCALL", "SELL", float(fcall_bid), fqty))
                actions.append(("FRUIT", "BUY", float(px["FRUIT"]["ask"]), hqty))
                return actions

    if fcall_ask is not None and px["FRUIT"]["bid"] is not None:
        if fcall_ask < FCALL_FAIR - THRESHOLD:
            fqty = cfg.base_size
            hqty = 10 * fqty
            if (
                _within_limits(pos["FCALL"], +fqty, cfg.max_pos)
                and _within_limits(pos["FRUIT"], -hqty, cfg.max_pos)
            ):
                actions.append(("FCALL", "BUY", float(fcall_ask), fqty))
                actions.append(("FRUIT", "SELL", float(px["FRUIT"]["bid"]), hqty))
                return actions

    # --- 3) Stat-arb (last priority) ---
    # IMPORTANT change: if we don't have BOTH sides for BOTH names, we do NOT trade.
    if not (_have_prices(px, "APPL") and _have_prices(px, "ORNG")):
        return []

    appl_mid = _mid(px["APPL"]["bid"], px["APPL"]["ask"])
    orng_mid = _mid(px["ORNG"]["bid"], px["ORNG"]["ask"])
    if appl_mid is None or orng_mid is None:
        return []

    fair_orng = MEAN_FAIR + CORR * (appl_mid - MEAN_FAIR)
    fair_appl = MEAN_FAIR + CORR * (orng_mid - MEAN_FAIR)

    # ORNG cheap -> buy ORNG (at ask), sell APPL (at bid)
    dev = fair_orng - float(px["ORNG"]["ask"])
    if dev > STAT_THRESHOLD:
        qty = cfg.base_size
        if _within_limits(pos["ORNG"], +qty, cfg.max_pos) and _within_limits(pos["APPL"], -qty, cfg.max_pos):
            actions.append(("ORNG", "BUY", float(px["ORNG"]["ask"]), qty))
            actions.append(("APPL", "SELL", float(px["APPL"]["bid"]), qty))
            return actions

    # ORNG expensive -> sell ORNG, buy APPL
    dev = float(px["ORNG"]["bid"]) - fair_orng
    if dev > STAT_THRESHOLD:
        qty = cfg.base_size
        if _within_limits(pos["ORNG"], -qty, cfg.max_pos) and _within_limits(pos["APPL"], +qty, cfg.max_pos):
            actions.append(("ORNG", "SELL", float(px["ORNG"]["bid"]), qty))
            actions.append(("APPL", "BUY", float(px["APPL"]["ask"]), qty))
            return actions

    # APPL cheap -> buy APPL, sell ORNG
    dev = fair_appl - float(px["APPL"]["ask"])
    if dev > STAT_THRESHOLD:
        qty = cfg.base_size
        if _within_limits(pos["APPL"], +qty, cfg.max_pos) and _within_limits(pos["ORNG"], -qty, cfg.max_pos):
            actions.append(("APPL", "BUY", float(px["APPL"]["ask"]), qty))
            actions.append(("ORNG", "SELL", float(px["ORNG"]["bid"]), qty))
            return actions

    # APPL expensive -> sell APPL, buy ORNG
    dev = float(px["APPL"]["bid"]) - fair_appl
    if dev > STAT_THRESHOLD:
        qty = cfg.base_size
        if _within_limits(pos["APPL"], -qty, cfg.max_pos) and _within_limits(pos["ORNG"], +qty, cfg.max_pos):
            actions.append(("APPL", "SELL", float(px["APPL"]["bid"]), qty))
            actions.append(("ORNG", "BUY", float(px["ORNG"]["ask"]), qty))
            return actions

    return []


# ---------- Execution / order management ----------

def _normalize_open_orders(orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Try to normalize order dict fields across possible schemas.
    Expected keys we use: side, price/px, quantity/qty, id/order_id.
    """
    out = []
    for o in orders or []:
        side = (o.get("side") or o.get("dir") or "").upper()
        px = o.get("price", o.get("px"))
        qty = o.get("quantity", o.get("qty"))
        oid = o.get("id", o.get("order_id"))
        out.append({"side": side, "price": _to_float(px), "qty": int(qty) if qty is not None else None, "id": oid})
    return out


def _should_refresh(prev_tick: int, cur_tick: int, cfg: ArbConfig) -> bool:
    return (cur_tick - prev_tick) >= cfg.refresh_ticks


def execute_actions(actions: List[Action]) -> None:
    for ticker, side, price, qty in actions:
        try:
            send_order(ticker, side, price, qty)
        except Exception as e:
            print(f"Error sending order for {ticker} {side} {price} {qty}: {e}")


def step(cfg: ArbConfig, state: Dict[str, Any]) -> None:
    if get_status() not in (None, "ACTIVE", "RUNNING"):
        return

    snap = observe(cfg)
    tick = snap.tick

    # Basic throttling: do not churn orders every loop
    last_tick = int(state.get("last_tick", -10**9))
    if not _should_refresh(last_tick, tick, cfg):
        return
    state["last_tick"] = tick

    actions = compute_actions(snap, cfg)

    # Periodically clear orders if we haven't acted in a while (stale cleanup)
    last_action_tick = int(state.get("last_action_tick", tick))
    if not actions and (tick - last_action_tick) >= cfg.cancel_if_no_action_ticks:
        for t in cfg.tickers:
            try:
                cancel_all_orders(t)
            except Exception as e:
                print(f"Error canceling orders for {t}: {e}")
        state["last_action_tick"] = tick
        return

    if not actions:
        return

    # If we do have actions, we cancel only the tickers we are about to touch.
    touched = sorted({t for (t, _, _, _) in actions})
    for t in touched:
        try:
            cancel_all_orders(t)
        except Exception as e:
            print(f"Error canceling orders for {t}: {e}")

    execute_actions(actions)
    state["last_action_tick"] = tick


def main(cfg: Optional[ArbConfig] = None) -> None:
    cfg = cfg or ArbConfig()
    state: Dict[str, Any] = {}
    while True:
        try:
            step(cfg, state)
        except Exception as e:
            print(f"Error in main loop: {e}")
        time.sleep(cfg.loop_sleep_s)


if __name__ == "__main__":
    main()
