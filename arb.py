from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from config import *

from utils import (
    cancel_order,
    cancel_all_orders,
    get_book,
    get_open_orders,
    get_position,
    get_status,
    get_tick,
    send_order,
)

TICK = 0.01
THRESHOLD = 0.1  # Threshold for arbitrage in dollars
STAT_THRESHOLD = 0.5  # Threshold for stat arb
MEAN_FAIR = 50.0
CORR = 0.82

@dataclass
class ArbConfig:
    tickers: List[str] = field(default_factory=lambda: ['FRUIT', 'APPL', 'ORNG'])
    loop_sleep_s: float = 0.25
    base_size: int = 100
    max_pos: int = 1000

@dataclass
class Snapshot:
    tick: int
    positions: Dict[str, int]
    books: Dict[str, Dict[str, Any]]
    open_orders: Dict[str, List[Dict[str, Any]]]

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

def observe(cfg: ArbConfig) -> Snapshot:
    tick = get_tick()
    positions = {ticker: int(get_position(ticker) or 0) for ticker in cfg.tickers}
    books = {ticker: get_book(ticker) for ticker in cfg.tickers}
    open_orders = {ticker: get_open_orders(ticker) or [] for ticker in cfg.tickers}
    return Snapshot(tick=tick, positions=positions, books=books, open_orders=open_orders)

def compute_actions(snap: Snapshot, cfg: ArbConfig) -> List[Tuple[str, str, Optional[float], int]]:
    actions = []
    books = snap.books
    positions = snap.positions

    # Get best prices
    prices = {}
    for ticker in cfg.tickers:
        bid, ask = _best_prices(books[ticker])
        prices[ticker] = {'bid': bid, 'ask': ask}

    # FRUIT arbitrage: FRUIT should = APPL + ORNG
    fruit_bid = prices['FRUIT']['bid']
    fruit_ask = prices['FRUIT']['ask']
    appl_bid = prices['APPL']['bid']
    appl_ask = prices['APPL']['ask']
    orng_bid = prices['ORNG']['bid']
    orng_ask = prices['ORNG']['ask']

    if fruit_bid is not None and appl_ask is not None and orng_ask is not None:
        synthetic_ask = appl_ask + orng_ask
        if fruit_bid > synthetic_ask + THRESHOLD:
            # Sell FRUIT, buy APPL, buy ORNG (limit orders at best prices for arb)
            if positions['FRUIT'] > -cfg.max_pos and positions['APPL'] < cfg.max_pos and positions['ORNG'] < cfg.max_pos:
                actions.append(('FRUIT', 'SELL', fruit_bid, cfg.base_size))  # Limit sell at best bid
                actions.append(('APPL', 'BUY', appl_ask, cfg.base_size))   # Limit buy at best ask
                actions.append(('ORNG', 'BUY', orng_ask, cfg.base_size))   # Limit buy at best ask

    if fruit_ask is not None and appl_bid is not None and orng_bid is not None:
        synthetic_bid = appl_bid + orng_bid
        if fruit_ask < synthetic_bid - THRESHOLD:
            # Buy FRUIT, sell APPL, sell ORNG (limit orders at best prices for arb)
            if positions['FRUIT'] < cfg.max_pos and positions['APPL'] > -cfg.max_pos and positions['ORNG'] > -cfg.max_pos:
                actions.append(('FRUIT', 'BUY', fruit_ask, cfg.base_size))  # Limit buy at best ask
                actions.append(('APPL', 'SELL', appl_bid, cfg.base_size))  # Limit sell at best bid
                actions.append(('ORNG', 'SELL', orng_bid, cfg.base_size))  # Limit sell at best bid

    # Statistical arbitrage on APPL and ORNG (limit orders)
    appl_mid = (prices['APPL']['bid'] + prices['APPL']['ask']) / 2 if prices['APPL']['bid'] and prices['APPL']['ask'] else MEAN_FAIR
    orng_mid = (prices['ORNG']['bid'] + prices['ORNG']['ask']) / 2 if prices['ORNG']['bid'] and prices['ORNG']['ask'] else MEAN_FAIR

    fair_orng = MEAN_FAIR + CORR * (appl_mid - MEAN_FAIR)
    fair_appl = MEAN_FAIR + CORR * (orng_mid - MEAN_FAIR)

    # If ORNG is cheap relative to APPL, buy ORNG sell APPL
    if prices['ORNG']['ask'] is not None and prices['APPL']['bid'] is not None:
        if fair_orng - prices['ORNG']['ask'] > STAT_THRESHOLD:
            if positions['ORNG'] < cfg.max_pos and positions['APPL'] > -cfg.max_pos:
                actions.append(('ORNG', 'BUY', prices['ORNG']['ask'], cfg.base_size))
                actions.append(('APPL', 'SELL', prices['APPL']['bid'], cfg.base_size))

    # If ORNG is expensive, sell ORNG buy APPL
    if prices['ORNG']['bid'] is not None and prices['APPL']['ask'] is not None:
        if prices['ORNG']['bid'] - fair_orng > STAT_THRESHOLD:
            if positions['ORNG'] > -cfg.max_pos and positions['APPL'] < cfg.max_pos:
                actions.append(('ORNG', 'SELL', prices['ORNG']['bid'], cfg.base_size))
                actions.append(('APPL', 'BUY', prices['APPL']['ask'], cfg.base_size))

    # Similarly for APPL
    if prices['APPL']['ask'] is not None and prices['ORNG']['bid'] is not None:
        if fair_appl - prices['APPL']['ask'] > STAT_THRESHOLD:
            if positions['APPL'] < cfg.max_pos and positions['ORNG'] > -cfg.max_pos:
                actions.append(('APPL', 'BUY', prices['APPL']['ask'], cfg.base_size))
                actions.append(('ORNG', 'SELL', prices['ORNG']['bid'], cfg.base_size))

    if prices['APPL']['bid'] is not None and prices['ORNG']['ask'] is not None:
        if prices['APPL']['bid'] - fair_appl > STAT_THRESHOLD:
            if positions['APPL'] > -cfg.max_pos and positions['ORNG'] < cfg.max_pos:
                actions.append(('APPL', 'SELL', prices['APPL']['bid'], cfg.base_size))
                actions.append(('ORNG', 'BUY', prices['ORNG']['ask'], cfg.base_size))

    return actions

def execute_actions(actions: List[Tuple[str, str, Optional[float], int]]) -> None:
    for ticker, side, price, qty in actions:
        try:
            send_order(ticker, side, price, qty)  # All limit orders now
        except Exception as e:
            print(f"Error sending order for {ticker} {side} {price} {qty}: {e}")

def step(cfg: ArbConfig) -> None:
    if get_status() not in (None, 'ACTIVE', 'RUNNING'):
        return
    # Cancel all open orders to avoid conflicts
    for ticker in cfg.tickers:
        try:
            cancel_all_orders(ticker)
        except Exception as e:
            print(f"Error canceling orders for {ticker}: {e}")
    snap = observe(cfg)
    actions = compute_actions(snap, cfg)
    execute_actions(actions)

def main(cfg: Optional[ArbConfig] = None) -> None:
    cfg = cfg or ArbConfig()
    while True:
        try:
            step(cfg)
        except Exception as e:
            print(f"Error in main loop: {e}")
        time.sleep(cfg.loop_sleep_s)

if __name__ == "__main__":
    main()