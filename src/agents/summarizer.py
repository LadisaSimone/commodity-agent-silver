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


def summarize(articles: list[dict], silver: dict, gold: dict) -> tuple[str, dict]:
    client = anthropic.Anthropic()
    today = date.today().strftime("%B %d, %Y")

    ratio = gold["price"] / silver["price"]
    metals_snapshot = (
        f"Silver (SI=F): {_fmt(silver)}\n"
        f"Gold   (GC=F): {_fmt(gold)}\n"
        f"Gold/Silver Ratio: {ratio:.1f}"
    )

    articles_text = "\n\n".join(
        f"Title: {a['title']}\nDate: {a['date']}\nURL: {a.get('url', '')}\nSummary: {a['description']}"
        for a in articles
    )

    prompt = _PROMPT_TEMPLATE.format(
        today=today,
        metals_snapshot=metals_snapshot,
        articles_text=articles_text,
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    return extract_scores(response.content[0].text)
