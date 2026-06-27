import yfinance as yf
from config.settings import TICKER


def fetch_silver_price() -> dict:
    fi = yf.Ticker(TICKER).fast_info
    price = fi.last_price
    prev_close = fi.previous_close
    change = price - prev_close
    change_pct = (change / prev_close) * 100
    return {"price": price, "change": change, "change_pct": change_pct}
