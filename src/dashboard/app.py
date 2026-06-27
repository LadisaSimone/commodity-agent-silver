import io
import re
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

load_dotenv(_root / ".env")

from src.fetchers.price import fetch_silver_price, fetch_gold_price
from src.fetchers.news import fetch_articles
from src.agents.summarizer import summarize
from config.settings import OUTPUTS_DIR


# ── helpers ───────────────────────────────────────────────────────────────────

def load_latest_briefing() -> tuple[str | None, str | None]:
    files = sorted(Path(OUTPUTS_DIR).glob("briefing_*.txt"))
    if not files:
        return None, None
    f = files[-1]
    return f.read_text(), f.stem.replace("briefing_", "")


@st.cache_data(ttl=300)
def cached_prices() -> tuple[dict, dict]:
    return fetch_silver_price(), fetch_gold_price()


def run_and_save() -> str:
    silver, gold = fetch_silver_price(), fetch_gold_price()
    articles = fetch_articles()
    briefing = summarize(articles, silver, gold)
    out_dir = Path(OUTPUTS_DIR)
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"briefing_{date.today().isoformat()}.txt").write_text(briefing)
    return briefing


def escape_dollars(text: str) -> str:
    """Prevent Streamlit from interpreting $ as LaTeX delimiters."""
    return text.replace("$", r"\$")


def add_section_dividers(text: str) -> str:
    """Insert --- before ALL-CAPS section headers (skips the first one)."""
    lines = text.splitlines()
    result = []
    for line in lines:
        stripped = line.strip()
        if (
            result
            and stripped
            and len(stripped) >= 8
            and re.match(r"^[A-Z][A-Z\s\/]+$", stripped)
        ):
            result.append("\n---\n")
        result.append(line)
    return "\n".join(result)


def split_conviction(text: str) -> tuple[str, str | None]:
    """Split the CONVICTION SCORE section from the rest of the briefing."""
    m = re.search(r"\n(CONVICTION SCORE\b.*)", text, re.DOTALL)
    if m:
        return text[: m.start()].strip(), m.group(1).strip()
    return text.strip(), None


def parse_score(conviction_text: str) -> float | None:
    m = re.search(r"Score:\s*(\d+(?:\.\d+)?)/10", conviction_text)
    return float(m.group(1)) if m else None


def score_badge_html(score: float) -> str:
    if score <= 3:
        bg, label = "#922b21", "BEARISH"
    elif score <= 6:
        bg, label = "#9a6d00", "NEUTRAL"
    else:
        bg, label = "#1a6b3a", "BULLISH"
    return (
        f'<span style="background:{bg};color:#fff;padding:6px 18px;'
        f"border-radius:20px;font-weight:700;font-size:1.05rem;"
        f'letter-spacing:0.06em;">{label}&ensp;{score}&thinsp;/&thinsp;10</span>'
    )


def generate_pdf(briefing: str, silver: dict, gold: dict, briefing_date: str) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import HRFlowable, SimpleDocTemplate, Paragraph, Spacer

    def md_to_rl(text: str) -> str:
        """Convert inline markdown to reportlab XML tags."""
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
        text = re.sub(r"[*_`]", "", text)  # strip leftover symbols
        return text

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=inch,
        rightMargin=inch,
    )
    styles = getSampleStyleSheet()
    story = []

    # Cover header
    story.append(Paragraph("Silver Market Intelligence Briefing", styles["h1"]))
    story.append(Paragraph(briefing_date, styles["Normal"]))
    story.append(Spacer(1, 6))

    ratio = gold["price"] / silver["price"]
    sign_s = "+" if silver["change"] >= 0 else ""
    sign_g = "+" if gold["change"] >= 0 else ""
    story.append(Paragraph(
        f"Silver: ${silver['price']:.2f} ({sign_s}{silver['change_pct']:.2f}%)  |  "
        f"Gold: ${gold['price']:,.2f} ({sign_g}{gold['change_pct']:.2f}%)  |  "
        f"Ratio: {ratio:.1f}",
        styles["Normal"],
    ))
    story.append(Spacer(1, 0.2 * inch))

    hr = lambda: HRFlowable(width="100%", thickness=0.5, color=colors.grey, spaceAfter=4)

    for line in briefing.split("\n"):
        stripped = line.strip()

        if not stripped:
            story.append(Spacer(1, 4))

        elif stripped == "---":
            story.append(hr())

        elif stripped.startswith("### "):
            story.append(Spacer(1, 6))
            story.append(Paragraph(md_to_rl(stripped[4:]), styles["h3"]))

        elif stripped.startswith("## "):
            story.append(Spacer(1, 8))
            story.append(Paragraph(md_to_rl(stripped[3:]), styles["h2"]))

        elif stripped.startswith("# "):
            story.append(Spacer(1, 8))
            story.append(Paragraph(md_to_rl(stripped[2:]), styles["h1"]))

        elif re.match(r"^[A-Z][A-Z\s\/]+$", stripped) and len(stripped) >= 8:
            # ALL-CAPS section header from Claude output
            story.append(Spacer(1, 10))
            story.append(hr())
            story.append(Paragraph(stripped, styles["h2"]))

        elif re.match(r"^[-*]\s", stripped):
            story.append(Paragraph(f"• {md_to_rl(stripped[2:])}", styles["Normal"]))

        elif re.match(r"^\d+\.\s", stripped):
            story.append(Paragraph(md_to_rl(stripped), styles["Normal"]))

        else:
            story.append(Paragraph(md_to_rl(stripped), styles["Normal"]))

    doc.build(story)
    return buffer.getvalue()


# ── theme CSS ─────────────────────────────────────────────────────────────────

_DOWNLOAD_BTN_CSS = """
[data-testid="stDownloadButton"] button {
    background-color: #1a73e8 !important;
    color: #ffffff !important;
    border: none !important;
}
[data-testid="stDownloadButton"] button:hover {
    background-color: #1557b0 !important;
    color: #ffffff !important;
}
"""

_DARK_CSS = f"""
<style>
.stApp, .main, .main .block-container {{ background-color: #0e1117 !important; }}
section[data-testid="stSidebar"] > div {{ background-color: #161b27 !important; }}
[data-testid="stMetricValue"] {{ color: #c0c0c0 !important; font-weight: 700; }}
{_DOWNLOAD_BTN_CSS}
</style>
"""

_LIGHT_CSS = f"""
<style>
.stApp, .main, .main .block-container {{ background-color: #f8f9fa !important; }}
section[data-testid="stSidebar"] > div {{ background-color: #eef0f4 !important; }}
h1, h2, h3, h4, h5, h6 {{ color: #0e1117 !important; }}
p, li, label, .stMarkdown {{ color: #31333f !important; }}
[data-testid="stMetricValue"] {{ color: #444444 !important; font-weight: 700; }}
[data-testid="stMetricLabel"] {{ color: #666666 !important; }}
[data-testid="stCaptionContainer"] {{ color: #666666 !important; }}
{_DOWNLOAD_BTN_CSS}
</style>
"""

# ── page config + CSS ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Silver Market Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

st.markdown(_DARK_CSS if st.session_state.dark_mode else _LIGHT_CSS, unsafe_allow_html=True)


# ── load data ─────────────────────────────────────────────────────────────────

silver, gold = cached_prices()
ratio = gold["price"] / silver["price"]


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.toggle("Dark mode", key="dark_mode")
    st.divider()
    st.markdown("### Silver Market Intelligence")
    st.markdown(
        "An AI-powered daily briefing agent that fetches live metals prices "
        "and the latest silver news, then generates a structured market "
        "analysis using Claude."
    )
    st.divider()
    st.markdown("**Data sources**")
    st.markdown("- Yahoo Finance (SI=F, GC=F)")
    st.markdown("- Google News RSS")
    st.markdown("- Claude Haiku 4.5")
    st.divider()
    last_updated_slot = st.empty()


# ── header ────────────────────────────────────────────────────────────────────

st.title("Silver Market Intelligence")
refresh = st.button("↻  Refresh", type="primary")
st.divider()


# ── metals snapshot ───────────────────────────────────────────────────────────

st.subheader("Metals Snapshot")

c1, c2, c3 = st.columns(3)
c1.metric(
    "Silver  (SI=F)",
    f"${silver['price']:.2f}",
    f"{silver['change']:+.2f}  ({silver['change_pct']:+.2f}%)",
)
c2.metric(
    "Gold  (GC=F)",
    f"${gold['price']:,.2f}",
    f"{gold['change']:+.2f}  ({gold['change_pct']:+.2f}%)",
)
c3.metric(
    "Gold / Silver Ratio",
    f"{ratio:.1f}",
)

st.divider()


# ── load / generate briefing ──────────────────────────────────────────────────

if refresh:
    cached_prices.clear()
    with st.spinner("Fetching prices, news, and generating briefing…"):
        briefing = run_and_save()
    briefing_date = date.today().isoformat()
else:
    briefing, briefing_date = load_latest_briefing()

last_updated_slot.markdown(f"**Last updated:** {briefing_date or '—'}")

if not briefing:
    st.info("No briefing found. Click **↻  Refresh** to generate one.")
    st.stop()


# ── conviction score (always visible above scrollable box) ────────────────────

body, conviction_section = split_conviction(briefing)
score = parse_score(conviction_section) if conviction_section else None

st.subheader("Conviction Score")
if score is not None:
    st.markdown(score_badge_html(score), unsafe_allow_html=True)
    st.write("")
if conviction_section:
    drivers = re.sub(r"^CONVICTION SCORE\s*\n", "", conviction_section)
    drivers = re.sub(r"Score:\s*\d+/10\s*\n?", "", drivers).strip()
    st.markdown(escape_dollars(drivers))

st.divider()


# ── daily briefing (scrollable) + download ────────────────────────────────────

col_header, col_dl = st.columns([5, 1])
with col_header:
    st.subheader("Daily Briefing")
    st.caption(f"Date: {briefing_date}")
with col_dl:
    st.write("")
    pdf_bytes = generate_pdf(briefing, silver, gold, briefing_date or "")
    st.download_button(
        "⬇ Download PDF",
        data=pdf_bytes,
        file_name=f"silver_briefing_{briefing_date}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

with st.container(height=520):
    st.markdown(add_section_dividers(escape_dollars(body)))
