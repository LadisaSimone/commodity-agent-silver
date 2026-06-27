import anthropic
from datetime import date
from pathlib import Path

from config.settings import MODEL, MAX_TOKENS

_PROMPT_TEMPLATE = (
    Path(__file__).parent.parent.parent / "prompts" / "briefing.txt"
).read_text()


def summarize(articles: list[dict], price: dict) -> str:
    client = anthropic.Anthropic()
    today = date.today().strftime("%B %d, %Y")

    sign = "+" if price["change"] >= 0 else ""
    price_line = (
        f"Live silver price (SI=F): ${price['price']:.2f}  "
        f"{sign}{price['change']:.2f} ({sign}{price['change_pct']:.2f}%)"
    )

    articles_text = "\n\n".join(
        f"Title: {a['title']}\nDate: {a['date']}\nSummary: {a['description']}"
        for a in articles
    )

    prompt = _PROMPT_TEMPLATE.format(
        today=today,
        price_line=price_line,
        articles_text=articles_text,
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text
