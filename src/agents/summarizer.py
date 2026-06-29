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


_DOMAIN_MAP = {
    "kitco":        "kitco.com",
    "fxempire":     "fxempire.com",
    "reuters":      "reuters.com",
    "cnbc":         "cnbc.com",
    "seekingalpha": "seekingalpha.com",
    "coindesk":     "coindesk.com",
    "bloomberg":    "bloomberg.com",
    "goldprice":    "goldprice.org",
    "investing":    "investing.com",
    "marketwatch":  "marketwatch.com",
    "wsj":          "wsj.com",
    "ft":           "ft.com",
    "zerohedge":    "zerohedge.com",
    "silverdoctors": "silverdoctors.com",
    "pvmagazine":    "pv-magazine.com",
    "pvmagazineusa": "pv-magazine-usa.com",
    "magazineusa":   "pv-magazine-usa.com",
}


def reattach_links(briefing: str, articles: list[dict]) -> str:
    article_urls = [a.get("url", "") for a in articles if a.get("url")]
    title_lookup = [
        (a["title"].lower(), a.get("url", ""))
        for a in articles
        if a.get("url")
    ]

    def find_url(source_name: str) -> str | None:
        # Strip trailing date suffix e.g. "KITCO, 26 Jun" → "KITCO"
        source_clean = re.sub(r',?\s*\d{1,2}\s+\w+$', '', source_name).strip()
        key = re.sub(r'\W+', '', source_clean.lower())
        source_lower = source_clean.lower()

        # Pass 1 — domain matching
        domain = _DOMAIN_MAP.get(key)
        if domain:
            for url in article_urls:
                if domain in url:
                    return url
        # Also try the raw key as a domain fragment if not in the map
        for url in article_urls:
            if key and key in url.lower():
                return url

        # Pass 2 — title keyword fallback (2+ word matches)
        words = [w for w in re.split(r'\W+', source_name.lower()) if len(w) > 2]
        best_url, best_count = None, 1
        for title, url in title_lookup:
            count = sum(1 for w in words if w in title)
            if count >= 2 and count > best_count:
                best_count = count
                best_url = url
        if best_url:
            return best_url

        # Pass 3 — source name substring in article title
        # Handles RSS patterns like "Gold & silver update — KITCO" where domain matching fails
        for article in articles:
            title = article.get("title", "").lower()
            if source_lower in title or key in title:
                return article.get("url", "")

        return None

    def replace_match(m: re.Match) -> str:
        source_name = m.group(1)
        url = find_url(source_name)
        if url:
            url = url.split('?')[0] if '?' in url else url
        return f"[{source_name}]({url})" if url else m.group(0)

    # Only replace bare [Text] — skip already-linked [Text](url)
    result = re.sub(r'\[([^\]]+)\](?!\()', replace_match, briefing)

    # Cleanup: remove (url) not preceded by ] — malformed link remnants
    result = re.sub(r'(?<!\])\(https?://[^)]+\)', '', result)
    # Remove any remaining bare URLs not in markdown link format
    result = re.sub(r'(?<!\()\bhttps?://\S+', '', result)

    return result


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

    def _try_parse(i: int) -> dict | None:
        line = lines[i].strip()
        if '"conviction"' not in line:
            return None
        try:
            data = json.loads(line[line.index("{"):])
            return {**_DEFAULT_SCORES, **data.get("conviction", {})}
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    # Search first 10 lines (JSON-first format)
    for i in range(min(10, len(lines))):
        scores = _try_parse(i)
        if scores is not None:
            # Remove the JSON line (and any immediately following blank line) from briefing
            rest_start = i + 1
            while rest_start < len(lines) and not lines[rest_start].strip():
                rest_start += 1
            return "\n".join(lines[rest_start:]).strip(), scores

    # Search last 10 lines (JSON-last format)
    search_start = max(0, len(lines) - 10)
    for i in range(len(lines) - 1, search_start - 1, -1):
        scores = _try_parse(i)
        if scores is not None:
            return "\n".join(lines[:i]).rstrip(), scores

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

    clean_briefing, scores = extract_scores(response.content[0].text)
    clean_briefing = reattach_links(clean_briefing, articles)
    return clean_briefing, scores
