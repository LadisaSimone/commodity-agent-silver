import anthropic
from datetime import date
from pathlib import Path

from config.settings import MODEL, MAX_TOKENS

_PROMPT_TEMPLATE = (
    Path(__file__).parent.parent.parent / "prompts" / "briefing.txt"
).read_text()


def _fmt(p: dict) -> str:
    sign = "+" if p["change"] >= 0 else ""
    return f"${p['price']:.2f}  {sign}{p['change']:.2f} ({sign}{p['change_pct']:.2f}%)"


def summarize(articles: list[dict], silver: dict, gold: dict) -> str:
    client = anthropic.Anthropic()
    today = date.today().strftime("%B %d, %Y")

    ratio = gold["price"] / silver["price"]
    metals_snapshot = (
        f"Silver (SI=F): {_fmt(silver)}\n"
        f"Gold   (GC=F): {_fmt(gold)}\n"
        f"Gold/Silver Ratio: {ratio:.1f}"
    )

    articles_text = "\n\n".join(
        f"Title: {a['title']}\nDate: {a['date']}\nSummary: {a['description']}"
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

    return response.content[0].text
