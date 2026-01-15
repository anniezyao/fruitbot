import requests
from config import *

s = requests.Session()


def _get(path, params=None):
    resp = s.get(API_URL + path, params=params)
    resp.raise_for_status()
    return resp.json()

def _post(path, params=None):
    resp = s.post(API_URL + path, params=params)
    resp.raise_for_status()
    return resp.json()

def _delete(path):
    resp = s.delete(API_URL + path)
    resp.raise_for_status()
    if resp.text:
        try:
            return resp.json()
        except Exception:
            return {}
    return {}

def get_case():
    return _get('case')

def get_tick():
    return get_case()['tick']

def get_status():
    return get_case().get('status')

def get_book(ticker, limit=None):
    params = {'ticker': ticker}
    if limit is not None:
        params['limit'] = limit
    return _get('securities/book', params=params)

def get_tas(ticker, limit):
    return _get('securities/tas', params={'ticker': ticker, 'limit': limit})

def get_securities(ticker=None):
    params = {}
    if ticker is not None:
        params['ticker'] = ticker
    return _get('securities', params=params if params else None)

def get_security(ticker):
    secs = get_securities(ticker=ticker)
    if isinstance(secs, list) and secs:
        return secs[0]
    return secs

def get_position(ticker):
    sec = get_security(ticker)
    return sec.get('position', 0)

def get_orders(ticker=None, status=None):
    params = {}
    if ticker is not None:
        params['ticker'] = ticker
    if status is not None:
        params['status'] = status
    return _get('orders', params=params if params else None)

def get_open_orders(ticker=None):
    return get_orders(ticker=ticker, status='OPEN')

def get_order(order_id):
    return _get('orders/' + str(order_id))

def send_order(ticker, side, price, size, order_type='LIMIT'):
    out = _post(
        'orders',
        params={
            'ticker': ticker,
            'type': order_type,
            'action': side,
            'quantity': size,
            **({} if price is None else {'price': price}),
        },
    )
    print('Sent', out.get('order_id')) 
    return out.get('order_id')

def cancel_order(order_id):
    print('Canceled', order_id)
    return _delete('orders/' + str(order_id))

def cancel_all_orders(ticker=None):
    params = {'all': 1}
    if ticker is not None:
        params['ticker'] = ticker
    return _post('commands/cancel', params=params)

def get_news(limit=None):
    params = {}
    if limit is not None:
        params['limit'] = limit
    return _get('news', params=params if params else None)
