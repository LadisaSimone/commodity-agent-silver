import anthropic
import json
from datetime import date
from pathlib import Path

from config.settings import MODEL, MAX_TOKENS

_PROMPT_TEMPLATE = (
    Path(__file__).parent.parent.parent / "prompts" / "briefing.txt"
).read_text()


def _fmt(p: dict) -> str:
    sign = "+" if p["change"] >= 0 else ""
    return f"${p['price']:.2f}  {sign}{p['change']:.2f} ({sign}{p['change_pct']:.2f}%)"


_DEFAULT_SCORES = {
    "macro": 5, "technicals": 5, "sentiment": 5,
    "etf_flows": 5, "industrial_demand": 5,
    "overall": 5, "verdict": "No conviction data available.", "supply_risk": "LOW",
}


def extract_scores(briefing_text: str) -> tuple[str, dict]:
    lines = briefing_text.splitlines()
    # Search the last 10 lines for the JSON block (Claude may pad with whitespace)
    search_start = max(0, len(lines) - 10)
    for i in range(len(lines) - 1, search_start - 1, -1):
        line = lines[i].strip()
        if '"conviction"' in line and line:
            try:
                # Handle any leading/trailing text around the JSON object
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
    significant_move: bool = False,
) -> tuple[str, dict]:
    client = anthropic.Anthropic()
    today = date.today().strftime("%B %d, %Y")

    def _sign(v: float) -> str:
        return "+" if v >= 0 else ""

    ratio = gold["price"] / silver["price"]
    metals_snapshot = (
        f"Silver (SI=F): {_fmt(silver)}\n"
        f"Gold   (GC=F): {_fmt(gold)}\n"
        f"Gold/Silver Ratio: {ratio:.1f}"
    )
    if dxy and dxy["price"]:
        metals_snapshot += (
            f"\nDXY (Dollar Index): {dxy['price']:.2f}"
            f"  {_sign(dxy['change'])}{dxy['change']:.2f}"
            f" ({_sign(dxy['change_pct'])}{dxy['change_pct']:.2f}%)"
        )
    if us10y and us10y["price"]:
        metals_snapshot += (
            f"\nUS 10Y Yield: {us10y['price']:.2f}%"
            f"  {_sign(us10y['change'])}{us10y['change']:.3f}"
            f" ({_sign(us10y['change_pct'])}{us10y['change_pct']:.2f}%)"
        )

    significant_move_context = ""
    if significant_move:
        sign = _sign(silver["change_pct"])
        significant_move_context = (
            f"\nNOTE: Silver is experiencing a significant intraday move "
            f"({sign}{silver['change_pct']:.2f}%). "
            f"Prioritize explaining the key drivers behind this move in your analysis."
        )

    articles_text = "\n\n".join(
        f"Title: {a['title']}\nDate: {a['date']}\nURL: {a.get('url', '')}\nSummary: {a['description']}"
        for a in articles
    )

    prompt = _PROMPT_TEMPLATE.format(
        today=today,
        metals_snapshot=metals_snapshot,
        significant_move_context=significant_move_context,
        articles_text=articles_text,
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    return extract_scores(response.content[0].text)
