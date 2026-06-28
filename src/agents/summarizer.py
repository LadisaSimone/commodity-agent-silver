import anthropic
import json
import re
import time
from datetime import date
from pathlib import Path

from config.settings import MODEL, MAX_TOKENS

_PROMPT_TEMPLATE = (
    Path(__file__).parent.parent.parent / "prompts" / "briefing.txt"
).read_text()

_DEFAULT_SCORES = {
    "macro": 5, "technicals": 5, "sentiment": 5,
    "etf_flows": 5, "industrial_demand": 5,
    "overall": 5, "verdict": "No conviction data available.", "supply_risk": "LOW",
}

_AVAILABLE_LABELS = {
    "silver_price": "Silver spot price & daily change",
    "gold_price": "Gold spot price & daily change",
    "ratio": "Gold/Silver Ratio",
    "dxy": "DXY (US Dollar Index)",
    "us10y": "US 10Y Yield",
    "rsi_14": "RSI-14 (computed from 30d history)",
    "volatility": "Volatility (30d annualized, computed)",
}
_PARTIAL_LABELS = {
    "etf_flows_proxy": "ETF Flow Data (SLV volume proxy only — not actual fund flows)",
}
_MISSING_LABELS = {
    "slv_actual_flows": "SLV Actual Flows (not available)",
    "comex_inventory": "COMEX Inventories (not available)",
    "cot_positioning": "COT Positioning (not available)",
    "open_interest": "Open Interest & Volume (not available)",
    "real_yields": "Real Yields (derived estimate only)",
}


def strip_urls(text: str) -> str:
    text = re.sub(r'\[([^\]]+)\]\(https?://[^\)]+\)', r'\1', text)
    text = re.sub(r'https?://\S+', '', text)
    return text


def _format_data_quality_block(data_quality: dict, today: str) -> str:
    lines = [f"DATA AVAILABILITY — {today}"]
    for k in data_quality.get("available", []):
        lines.append(f"✅ {_AVAILABLE_LABELS.get(k, k)}")
    for k in data_quality.get("partial", []):
        lines.append(f"⚠ {_PARTIAL_LABELS.get(k, k)}")
    for k in data_quality.get("missing", []):
        lines.append(f"❌ {_MISSING_LABELS.get(k, k)}")
    lines.append("")
    reliability = data_quality.get("reliability", "MEDIUM")
    reason = data_quality.get("reliability_reason", "Core price data available").lower()
    lines.append(f"Analysis reliability today: {reliability} — {reason}")
    return "\n".join(lines)


def _build_fallback_signals(silver: dict, gold: dict, dxy: dict | None, us10y: dict | None) -> str:
    """Minimal signals text for callers that don't pass a pre-computed signals block."""
    def _sign(v: float) -> str:
        return "+" if v >= 0 else ""

    ratio = gold["price"] / silver["price"]
    lines = [
        "QUANTITATIVE SIGNALS (pre-computed)",
        f"Silver: ${silver['price']:.2f} | {_sign(silver['change_pct'])}{silver['change_pct']:.2f}%",
        f"Gold: ${gold['price']:.2f} | {_sign(gold['change_pct'])}{gold['change_pct']:.2f}%",
        f"Ratio: {ratio:.1f}",
    ]
    if dxy and dxy.get("price"):
        lines.append(
            f"DXY: {dxy['price']:.2f} | {_sign(dxy['change_pct'])}{dxy['change_pct']:.2f}%"
        )
    if us10y and us10y.get("price"):
        lines.append(
            f"US10Y: {us10y['price']:.2f}% | {_sign(us10y['change_pct'])}{us10y['change_pct']:.2f}%"
        )
    return "\n".join(lines)


def extract_scores(briefing_text: str) -> tuple[str, dict]:
    lines = briefing_text.splitlines()
    search_start = max(0, len(lines) - 10)
    for i in range(len(lines) - 1, search_start - 1, -1):
        line = lines[i].strip()
        if '"conviction"' in line and line:
            try:
                json_start = line.index("{")
                data = json.loads(line[json_start:])
                scores = {**_DEFAULT_SCORES, **data.get("conviction", {})}
                clean_text = "\n".join(lines[:i]).rstrip()
                return clean_text, scores
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return briefing_text, dict(_DEFAULT_SCORES)


def summarize(
    articles: list[dict],
    silver: dict,
    gold: dict,
    dxy: dict | None = None,
    us10y: dict | None = None,
    signals_text: str | None = None,
    data_quality: dict | None = None,
) -> tuple[str, dict]:
    client = anthropic.Anthropic()
    today = date.today().strftime("%B %d, %Y")

    quantitative_signals = signals_text or _build_fallback_signals(
        silver, gold, dxy, us10y
    )

    data_quality_block = (
        _format_data_quality_block(data_quality, today)
        if data_quality is not None
        else f"DATA AVAILABILITY — {today}\n✅ Silver spot price & daily change\n✅ Gold spot price & daily change\n✅ Gold/Silver Ratio\n\nAnalysis reliability today: MEDIUM"
    )

    articles_text = strip_urls("\n\n".join(
        f"Title: {a['title']}\nDate: {a['date']}\nSource: {a.get('url', '')}\nSummary: {a['description']}"
        for a in articles
    ))

    prompt = _PROMPT_TEMPLATE.format(
        today=today,
        quantitative_signals=quantitative_signals,
        data_quality=data_quality_block,
        articles_text=articles_text,
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
                timeout=120.0,
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"API error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    return extract_scores(response.content[0].text)
