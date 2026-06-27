# Silver Market Intelligence Agent

A Python CLI agent that fetches live silver market data and generates a daily AI-powered briefing using Claude.

## Features

- Live silver spot price via Yahoo Finance (`SI=F`)
- Latest news scraped from Google News RSS
- AI briefing with ranked top stories, market analysis, and a daily conviction score
- Briefings saved automatically to `outputs/`

## Project Structure

```
commodity-agent-silver/
├── main.py                   # Entry point
├── requirements.txt
├── .env                      # API keys (not committed)
├── config/
│   └── settings.py           # Model, ticker, and app configuration
├── prompts/
│   └── briefing.txt          # Prompt template for Claude
├── src/
│   ├── fetchers/
│   │   ├── price.py          # Live silver price via yfinance
│   │   └── news.py           # Google News RSS scraping
│   ├── agents/
│   │   └── summarizer.py     # Claude API summarization
│   └── dashboard/
│       └── app.py            # Streamlit dashboard (coming soon)
├── outputs/                  # Saved daily briefings
└── tests/                    # Test suite (coming soon)
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root:
   ```
   ANTHROPIC_API_KEY=your_key_here
   ```

## Usage

Run the terminal briefing:
```bash
python main.py
```

Launch the Streamlit dashboard (coming soon):
```bash
streamlit run src/dashboard/app.py
```

Each briefing is saved to `outputs/briefing_YYYY-MM-DD.txt`.

## Briefing Structure

1. **TOP STORIES BY IMPACT** — top 5 articles ranked by market relevance
2. **Market Analysis** — price movements, drivers, notable news, outlook
3. **CONVICTION SCORE** — bullish/bearish score (1–10) with top 3 drivers
