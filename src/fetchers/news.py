import requests
from bs4 import BeautifulSoup
from config.settings import RSS_URL, MAX_ARTICLES


def fetch_articles() -> list[dict]:
    response = requests.get(
        RSS_URL,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "xml")
    articles = []
    for item in soup.find_all("item")[:MAX_ARTICLES]:
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
