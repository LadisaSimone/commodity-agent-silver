import io
import re
import sys
from datetime import date
from pathlib import Path

import anthropic
from dotenv import load_dotenv
import streamlit as st

_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

load_dotenv(_root / ".env")

from src.fetchers.price import fetch_silver_price, fetch_gold_price
from src.fetchers.news import fetch_articles
from src.agents.summarizer import summarize
from config.settings import MODEL, OUTPUTS_DIR


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
    return text.replace("$", r"\$")


def add_section_dividers(text: str) -> str:
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
    m = re.search(r"\n(CONVICTION SCORE\b.*)", text, re.DOTALL)
    if m:
        return text[: m.start()].strip(), m.group(1).strip()
    return text.strip(), None


def parse_score(conviction_text: str) -> float | None:
    m = re.search(r"Score:\s*(\d+(?:\.\d+)?)/10", conviction_text)
    return float(m.group(1)) if m else None


def parse_components(conviction_text: str) -> dict[str, float]:
    patterns = {
        "Macro": r"Macro:\s*([\d.]+)/10",
        "Technicals": r"Technicals:\s*([\d.]+)/10",
        "Sentiment": r"Sentiment:\s*([\d.]+)/10",
        "ETF Flows": r"ETF Flows:\s*([\d.]+)/10",
        "Industrial": r"Industrial Demand:\s*([\d.]+)/10",
    }
    return {
        name: float(m.group(1))
        for name, pat in patterns.items()
        if (m := re.search(pat, conviction_text))
    }


def price_widget_html(label: str, price_str: str, change: float, change_pct: float) -> str:
    color = "#38a169" if change >= 0 else "#e53e3e"
    sign = "+" if change >= 0 else ""
    return (
        f'<div style="font-size:0.65rem;color:#5a6a7e;font-weight:600;'
        f'text-transform:uppercase;letter-spacing:0.09em;margin-bottom:5px;">{label}</div>'
        f'<div style="font-size:1.9rem;font-weight:700;color:#d4e0f0;'
        f'font-family:\'SF Mono\',\'Fira Mono\',monospace;line-height:1.1;">{price_str}</div>'
        f'<div style="font-size:0.88rem;color:{color};font-weight:600;margin-top:3px;">'
        f'{sign}{change:.2f}&nbsp;&nbsp;({sign}{change_pct:.2f}%)</div>'
    )


def score_badge_html(score: float) -> str:
    if score <= 3:
        bg, label = "#7a1c1c", "BEARISH"
    elif score <= 6:
        bg, label = "#6b4400", "NEUTRAL"
    else:
        bg, label = "#145230", "BULLISH"
    return (
        f'<span style="background:{bg};color:#e8e8e8;padding:5px 14px;'
        f'border-radius:3px;font-weight:700;font-size:0.82rem;'
        f'letter-spacing:0.1em;white-space:nowrap;'
        f'font-family:\'SF Mono\',monospace;">'
        f'{label}&ensp;{score}/10</span>'
    )


def conviction_row_html(score: float | None, components: dict[str, float]) -> str:
    if score is None:
        return ""

    def bar_color(v: float) -> str:
        if v <= 3:
            return "#c53030"
        if v <= 6:
            return "#b7791f"
        return "#276749"

    bars = ""
    for name, val in components.items():
        pct = val / 10 * 100
        color = bar_color(val)
        bars += (
            '<div style="flex:1;min-width:75px;">'
            '<div style="display:flex;justify-content:space-between;align-items:baseline;'
            'margin-bottom:5px;">'
            f'<span style="font-size:0.62rem;color:#5a6a7e;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:0.07em;">{name}</span>'
            f'<span style="font-size:0.75rem;font-weight:700;color:#c8d4e8;'
            f'font-family:monospace;">{val}/10</span>'
            '</div>'
            '<div style="background:#182030;height:4px;border-radius:2px;overflow:hidden;">'
            f'<div style="background:{color};width:{pct:.0f}%;height:100%;border-radius:2px;"></div>'
            '</div>'
            '</div>'
        )

    return (
        '<div style="display:flex;align-items:center;gap:18px;padding:12px 18px;'
        'background:#0c1120;border:1px solid #1a2540;border-radius:5px;">'
        f'<div style="flex-shrink:0;">{score_badge_html(score)}</div>'
        '<div style="width:1px;height:34px;background:#1a2540;flex-shrink:0;"></div>'
        f'<div style="flex:1;display:flex;gap:14px;align-items:center;">{bars}</div>'
        '</div>'
    )


def generate_pdf(briefing: str, silver: dict, gold: dict, briefing_date: str) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import HRFlowable, SimpleDocTemplate, Paragraph, Spacer

    def md_to_rl(text: str) -> str:
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
        text = re.sub(r"[*_`]", "", text)
        return text

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=inch, rightMargin=inch,
    )
    styles = getSampleStyleSheet()
    story = []

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
            story.append(Spacer(1, 10))
            story.append(hr())
            story.append(Paragraph(stripped, styles["h2"]))
        elif re.match(r"^[-*]\s", stripped):
            story.append(Paragraph(f"• {md_to_rl(stripped[2:])}", styles["Normal"]))
        else:
            story.append(Paragraph(md_to_rl(stripped), styles["Normal"]))

    doc.build(story)
    return buffer.getvalue()


def ask_analyst(briefing: str | None, messages: list[dict]) -> str:
    client = anthropic.Anthropic()
    system = (
        "You are a concise silver market analyst. Answer questions about the silver "
        "market accurately, drawing on today's briefing where relevant. "
        "Keep answers under 150 words."
    )
    if briefing:
        system += f"\n\nToday's market briefing:\n\n{briefing}"
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system,
        messages=messages,
    )
    return response.content[0].text


# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
<style>
/* base */
.stApp, .main, .main .block-container {
    background-color: #080d18 !important;
    padding-top: 0.6rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}
section[data-testid="stSidebar"] > div {
    background-color: #0c1120 !important;
    border-right: 1px solid #1a2540 !important;
}

/* typography */
h1, h2, h3, h4 { color: #d4e0f0 !important; letter-spacing: -0.01em; }
p, li, .stMarkdown p { color: #8a9ab5 !important; }
label { color: #5a6a7e !important; }

/* sidebar */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] .stMarkdown p {
    color: #4a5a72 !important;
    font-size: 0.8rem !important;
}
section[data-testid="stSidebar"] strong { color: #5a6a7e !important; }
section[data-testid="stSidebar"] h3 {
    color: #6a7a8e !important;
    font-size: 0.88rem !important;
}

/* dividers */
hr { border-color: #1a2540 !important; margin: 0.4rem 0 !important; }

/* form input */
[data-testid="stTextInput"] input {
    background-color: #0c1120 !important;
    color: #c8d4e8 !important;
    border: 1px solid #1e2f4a !important;
    border-radius: 3px !important;
    font-size: 0.82rem !important;
}
[data-testid="stTextInput"] input::placeholder { color: #2d3f5a !important; }

/* Ask → button */
[data-testid="stFormSubmitButton"] button {
    background-color: #0c1120 !important;
    color: #d4af37 !important;
    border: 1px solid #d4af37 !important;
    border-radius: 3px !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
}
[data-testid="stFormSubmitButton"] button:hover {
    background-color: #d4af37 !important;
    color: #080d18 !important;
}

/* Refresh (primary) */
button[kind="primary"] {
    background-color: #d4af37 !important;
    color: #080d18 !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    border-radius: 3px !important;
}
button[kind="primary"]:hover { background-color: #e6c84a !important; }

/* secondary (clear history) */
button[kind="secondary"] {
    background-color: transparent !important;
    color: #2d3f5a !important;
    border: 1px solid #1a2540 !important;
    font-size: 0.7rem !important;
    border-radius: 3px !important;
}
button[kind="secondary"]:hover {
    color: #6a7a8e !important;
    border-color: #2d3f5a !important;
}

/* download */
[data-testid="stDownloadButton"] button {
    background-color: #132040 !important;
    color: #7aa8e0 !important;
    border: 1px solid #1e3460 !important;
    font-weight: 600 !important;
    border-radius: 3px !important;
    font-size: 0.82rem !important;
}
[data-testid="stDownloadButton"] button:hover {
    background-color: #1a2f57 !important;
    color: #a0c0f0 !important;
}

/* scrollable containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #1a2540 !important;
    border-radius: 4px !important;
}

/* chat messages */
[data-testid="stChatMessage"] {
    background-color: #0c1120 !important;
    border-radius: 3px !important;
}

/* caption */
[data-testid="stCaptionContainer"] { color: #2d3f5a !important; }
</style>
"""

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Silver Market Intelligence",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown(_CSS, unsafe_allow_html=True)

# ── prices ────────────────────────────────────────────────────────────────────

silver, gold = cached_prices()
ratio = gold["price"] / silver["price"]

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🪙 Silver Market Intelligence")
    st.markdown(
        "AI-powered daily briefing: live metals prices and the latest "
        "silver news analyzed by Claude."
    )
    st.divider()
    st.markdown("**Data sources**")
    st.markdown("- Yahoo Finance (SI=F, GC=F)")
    st.markdown("- Google News RSS")
    st.divider()
    last_updated_slot = st.empty()

# ── header row ────────────────────────────────────────────────────────────────

hdr_col, btn_col = st.columns([7, 1])
with hdr_col:
    st.markdown("## 🪙 Silver Market Intelligence")
with btn_col:
    refresh = st.button("↻ Refresh", type="primary", use_container_width=True)

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
    st.info("No briefing found. Click **↻ Refresh** to generate one.")
    st.stop()

body, conviction_section = split_conviction(briefing)
score = parse_score(conviction_section) if conviction_section else None
components = parse_components(conviction_section) if conviction_section else {}

# ── row 1: silver | gold + ratio | chat ───────────────────────────────────────

col_silver, col_gold, col_chat = st.columns([1, 1, 2])

with col_silver:
    st.markdown(
        price_widget_html("Silver (SI=F)", f"${silver['price']:.2f}", silver["change"], silver["change_pct"]),
        unsafe_allow_html=True,
    )

with col_gold:
    st.markdown(
        price_widget_html("Gold (GC=F)", f"${gold['price']:,.2f}", gold["change"], gold["change_pct"]),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="margin-top:12px;">'
        '<div style="font-size:0.65rem;color:#5a6a7e;font-weight:600;'
        'text-transform:uppercase;letter-spacing:0.09em;margin-bottom:5px;">Gold / Silver Ratio</div>'
        f'<div style="font-size:1.5rem;font-weight:700;color:#d4af37;'
        f'font-family:\'SF Mono\',monospace;">{ratio:.1f}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

with col_chat:
    st.markdown(
        '<div style="font-size:0.65rem;color:#5a6a7e;font-weight:600;'
        'text-transform:uppercase;letter-spacing:0.09em;margin-bottom:6px;">'
        'Ask the Silver Analyst</div>',
        unsafe_allow_html=True,
    )
    with st.container(height=180):
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    with st.form("analyst_form", clear_on_submit=True):
        qi, bi = st.columns([5, 1])
        with qi:
            user_question = st.text_input(
                "Question",
                placeholder="Ask me anything about silver…",
                label_visibility="collapsed",
            )
        with bi:
            submitted = st.form_submit_button("Ask →", use_container_width=True)
    if st.session_state.chat_history:
        if st.button("✕ clear history", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

if submitted and user_question.strip():
    q = user_question.strip()
    st.session_state.chat_history.append({"role": "user", "content": q})
    with st.spinner("Thinking…"):
        answer = ask_analyst(briefing, st.session_state.chat_history)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.rerun()

st.divider()

# ── row 2: conviction score ───────────────────────────────────────────────────

st.markdown(conviction_row_html(score, components), unsafe_allow_html=True)

st.divider()

# ── row 3: daily briefing ─────────────────────────────────────────────────────

brf_col, dl_col = st.columns([6, 1])
with brf_col:
    st.markdown(
        '<div style="font-size:0.65rem;color:#5a6a7e;font-weight:600;'
        'text-transform:uppercase;letter-spacing:0.09em;">'
        f'Daily Briefing'
        f'<span style="color:#2d3f5a;font-weight:400;margin-left:12px;">{briefing_date}</span>'
        '</div>',
        unsafe_allow_html=True,
    )
with dl_col:
    pdf_bytes = generate_pdf(briefing, silver, gold, briefing_date or "")
    st.download_button(
        "⬇ PDF",
        data=pdf_bytes,
        file_name=f"silver_briefing_{briefing_date}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

with st.container(height=450):
    st.markdown(add_section_dividers(escape_dollars(body)))
