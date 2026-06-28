from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from src.fetchers.price import fetch_silver_price, fetch_gold_price
from src.fetchers.news import fetch_articles
from src.agents.summarizer import summarize
from config.settings import OUTPUTS_DIR

load_dotenv()


def _metals_snapshot(silver: dict, gold: dict) -> str:
    def fmt(p):
        sign = "+" if p["change"] >= 0 else ""
        return f"${p['price']:.2f}  {sign}{p['change']:.2f}  ({sign}{p['change_pct']:.2f}%)"

    ratio = gold["price"] / silver["price"]
    return (
        f"  Silver  (SI=F)  {fmt(silver)}\n"
        f"  Gold    (GC=F)  {fmt(gold)}\n"
        f"  Gold/Silver Ratio: {ratio:.1f}"
    )


def save_briefing(briefing: str) -> Path:
    out_dir = Path(OUTPUTS_DIR)
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"briefing_{date.today().isoformat()}.txt"
    path.write_text(briefing)
    return path


def print_briefing(briefing: str, silver: dict, gold: dict) -> None:
    today = date.today().strftime("%B %d, %Y")
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  SILVER MARKET INTELLIGENCE BRIEFING")
    print(f"  {today}")
    print(f"{sep}")
    print(f"\n  METALS SNAPSHOT")
    print(_metals_snapshot(silver, gold))
    print(f"\n{sep}\n")
    print(briefing)
    print(f"\n{sep}\n")


def main() -> None:
    print("Fetching live metals prices...")
    silver = fetch_silver_price()
    gold = fetch_gold_price()
    print("Fetching silver news from Google News RSS...")
    articles = fetch_articles()
    print(f"Found {len(articles)} articles. Generating briefing...")
    briefing, scores = summarize(articles, silver, gold)
    print_briefing(briefing, silver, gold)
    print("Scores:", scores)
    path = save_briefing(briefing)
    print(f"Briefing saved to {path}")


if __name__ == "__main__":
    main()
