import yfinance as yf
from config.settings import TICKER, GOLD_TICKER


def fetch_price(ticker: str) -> dict:
    fi = yf.Ticker(ticker).fast_info
    price = fi.last_price
    prev_close = fi.previous_close
    change = price - prev_close
    change_pct = (change / prev_close) * 100
    return {"ticker": ticker, "price": price, "change": change, "change_pct": change_pct}


def fetch_silver_price() -> dict:
    return fetch_price(TICKER)


def fetch_gold_price() -> dict:
    return fetch_price(GOLD_TICKER)
