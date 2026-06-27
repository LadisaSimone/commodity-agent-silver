import anthropic
from datetime import date
from dotenv import load_dotenv

import requests
from bs4 import BeautifulSoup

load_dotenv()

RSS_URL = (
    "https://news.google.com/rss/search"
    "?q=silver+market+price&hl=en-US&gl=US&ceid=US:en"
)


def fetch_articles() -> list[dict]:
    response = requests.get(
        RSS_URL,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "xml")
    articles = []
    for item in soup.find_all("item")[:15]:
        title = item.find("title")
        description = item.find("description")
        pub_date = item.find("pubDate")

        raw_desc = description.get_text() if description else ""
        clean_desc = BeautifulSoup(raw_desc, "html.parser").get_text(strip=True)

        articles.append({
            "title": title.get_text(strip=True) if title else "",
            "date": pub_date.get_text(strip=True) if pub_date else "",
            "description": clean_desc,
        })

    return articles


def summarize(articles: list[dict]) -> str:
    client = anthropic.Anthropic()
    today = date.today().strftime("%B %d, %Y")

    articles_text = "\n\n".join(
        f"Title: {a['title']}\nDate: {a['date']}\nSummary: {a['description']}"
        for a in articles
    )

    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": f"""You are a silver market intelligence analyst. Today is {today}.

Here are the latest silver-related news headlines scraped from Google News:

{articles_text}

Based on these articles, produce a clean daily briefing covering:
1. Current silver price and recent price movements (if mentioned)
2. Key market drivers and factors affecting silver today
3. Notable news stories about silver (industrial demand, investment demand, mining)
4. Brief market outlook

Then add a final section titled "CONVICTION SCORE" containing:
- A single integer score from 1 to 10 (1 = very bearish, 10 = very bullish) on its own line in the format: Score: X/10
- The top 3 drivers behind the score, each as a short numbered bullet

Format the briefing clearly with sections and make it concise and actionable.""",
            }
        ],
    )

    return response.content[0].text


def print_briefing(briefing: str) -> None:
    today = date.today().strftime("%B %d, %Y")
    separator = "=" * 60
    print(f"\n{separator}")
    print(f"  SILVER MARKET INTELLIGENCE BRIEFING")
    print(f"  {today}")
    print(f"{separator}\n")
    print(briefing)
    print(f"\n{separator}\n")


def main() -> None:
    print("Fetching silver news from Google News RSS...")
    articles = fetch_articles()
    print(f"Found {len(articles)} articles. Generating briefing...")
    briefing = summarize(articles)
    print_briefing(briefing)


if __name__ == "__main__":
    main()
