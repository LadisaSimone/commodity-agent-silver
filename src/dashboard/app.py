import io
import json
import re
import sys
from datetime import date
from pathlib import Path

import anthropic
from dotenv import load_dotenv
import plotly.graph_objects as go
import streamlit as st

_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

load_dotenv(_root / ".env")

from src.fetchers.price import (
    fetch_silver_price, fetch_gold_price,
    fetch_dxy_price, fetch_us10y_price,
    fetch_silver_history,
)
from src.fetchers.news import fetch_articles
from src.agents.summarizer import summarize, extract_scores
from config.settings import MODEL, OUTPUTS_DIR


# ── helpers ───────────────────────────────────────────────────────────────────

def load_latest_briefing() -> tuple[str | None, str | None, dict]:
    files = sorted(Path(OUTPUTS_DIR).glob("briefing_*.txt"))
    if not files:
        return None, None, {}
    f = files[-1]
    date_str = f.stem.replace("briefing_", "")
    raw = f.read_text()
    briefing_text, scores = extract_scores(raw)
    # Prefer the companion scores file if present (more reliable than text extraction)
    scores_file = Path(OUTPUTS_DIR) / f"scores_{date_str}.json"
    if scores_file.exists():
        try:
            scores = json.loads(scores_file.read_text())
        except Exception:
            pass
    return briefing_text, date_str, scores


@st.cache_data(ttl=300)
def cached_prices() -> tuple[dict, dict, dict, dict]:
    return fetch_silver_price(), fetch_gold_price(), fetch_dxy_price(), fetch_us10y_price()


@st.cache_data(ttl=3600)
def cached_silver_history() -> list[dict]:
    return fetch_silver_history(30)


def run_and_save() -> tuple[str, dict]:
    silver, gold, dxy, us10y = (
        fetch_silver_price(), fetch_gold_price(), fetch_dxy_price(), fetch_us10y_price()
    )
    articles = fetch_articles()
    significant_move = abs(silver["change_pct"]) >= 2.0
    briefing_text, scores = summarize(
        articles, silver, gold, dxy=dxy, us10y=us10y, significant_move=significant_move
    )
    out_dir = Path(OUTPUTS_DIR)
    out_dir.mkdir(exist_ok=True)
    today = date.today().isoformat()
    (out_dir / f"briefing_{today}.txt").write_text(briefing_text)
    (out_dir / f"scores_{today}.json").write_text(json.dumps(scores))
    return briefing_text, scores


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


def score_color(v: int | float) -> str:
    if v <= 4:
        return "#e53e3e"
    if v <= 6:
        return "#d4af37"
    return "#38a169"


def colored_label_html(label: str, score: int | float) -> str:
    color = score_color(score)
    return (
        f'<span style="color:{color};font-size:0.72rem;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:0.06em;">{label}</span>'
    )


def supply_risk_badge_html(level: str) -> str:
    colors_map = {"LOW": "#145230", "MEDIUM": "#6b4400", "HIGH": "#7a1c1c"}
    bg = colors_map.get(level.upper(), "#1a2540")
    return (
        f'<span style="background:{bg};color:#e8e8e8;padding:3px 12px;'
        f'border-radius:3px;font-weight:700;font-size:0.78rem;'
        f'letter-spacing:0.1em;font-family:\'SF Mono\',monospace;">'
        f'{level.upper()}</span>'
    )


def overall_badge_html(score: int | float) -> str:
    if score <= 4:
        label, bg = "BEARISH", "#7a1c1c"
    elif score <= 6:
        label, bg = "NEUTRAL", "#6b4400"
    else:
        label, bg = "BULLISH", "#145230"
    return (
        f'<span style="background:{bg};color:#e8e8e8;padding:4px 14px;'
        f'border-radius:3px;font-weight:700;font-size:0.8rem;'
        f'letter-spacing:0.1em;font-family:\'SF Mono\',monospace;">'
        f'{label}</span>'
    )


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


def macro_widget_html(items: list[tuple[str, str, float, float]]) -> str:
    """Render two compact stacked macro metrics (label, value_str, change, change_pct)."""
    parts = []
    for label, value_str, change, change_pct in items:
        color = "#38a169" if change >= 0 else "#e53e3e"
        sign = "+" if change >= 0 else ""
        parts.append(
            f'<div style="margin-bottom:10px;">'
            f'<div style="font-size:0.6rem;color:#5a6a7e;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:0.09em;margin-bottom:2px;">{label}</div>'
            f'<div style="font-size:1.3rem;font-weight:700;color:#d4e0f0;'
            f'font-family:\'SF Mono\',monospace;line-height:1.1;">{value_str}</div>'
            f'<div style="font-size:0.78rem;color:{color};font-weight:600;">'
            f'{sign}{change:.2f}&nbsp;({sign}{change_pct:.2f}%)</div>'
            f'</div>'
        )
    return "".join(parts)


def render_silver_chart(history: list[dict]) -> go.Figure:
    dates = [h["date"] for h in history]
    closes = [h["close"] for h in history]
    high_val, low_val = max(closes), min(closes)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=closes,
        mode="lines",
        line=dict(color="#d4af37", width=2),
        hovertemplate="<b>%{x}</b><br>$%{y:.2f}<extra></extra>",
    ))
    fig.add_hline(
        y=high_val, line_dash="dash", line_color="rgba(56,161,105,0.45)", line_width=1,
        annotation_text=f"30d hi ${high_val:.2f}",
        annotation_font=dict(color="#38a169", size=9),
        annotation_position="top right",
    )
    fig.add_hline(
        y=low_val, line_dash="dash", line_color="rgba(229,62,62,0.45)", line_width=1,
        annotation_text=f"30d lo ${low_val:.2f}",
        annotation_font=dict(color="#e53e3e", size=9),
        annotation_position="bottom right",
    )
    fig.update_layout(
        title=dict(text="30-day silver price (SI=F)", font=dict(color="#5a6a7e", size=10), x=0, pad=dict(l=0)),
        paper_bgcolor="#080d18",
        plot_bgcolor="#080d18",
        margin=dict(t=28, b=8, l=4, r=4),
        height=180,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showline=False,
                   tickfont=dict(color="#2d3f5a", size=9)),
        yaxis=dict(showgrid=False, zeroline=False, showline=False,
                   tickfont=dict(color="#2d3f5a", size=9), tickprefix="$"),
    )
    return fig


def extract_top_stories(text: str) -> str:
    m = re.search(r"TOP STORIES BY IMPACT[^\n]*\n(.*?)(?=\n[A-Z][A-Z\s\/]{7,}|\Z)", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_bull_bear(text: str) -> tuple[str, str]:
    m = re.search(r"BULL VS BEAR[^\n]*\n(.*?)(?=\n[A-Z][A-Z\s\/]{7,}|\Z)", text, re.DOTALL)
    if not m:
        return "", ""
    section = m.group(1)
    bull_m = re.search(r"BULLISH CASE\n(.*?)(?=BEARISH CASE|\Z)", section, re.DOTALL)
    bear_m = re.search(r"BEARISH CASE\n(.*?)(?=VERDICT:|\Z)", section, re.DOTALL)
    return (
        bull_m.group(1).strip() if bull_m else "",
        bear_m.group(1).strip() if bear_m else "",
    )


def extract_supply_risk_reason(text: str) -> str:
    m = re.search(r"Risk:\s*(?:LOW|MEDIUM|HIGH)[^\n]*\n([^\n]+)", text)
    return m.group(1).strip() if m else "No supply disruptions detected."


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

silver, gold, dxy, us10y = cached_prices()
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
    st.markdown("- Yahoo Finance (SI=F, GC=F, DX-Y.NYB, ^TNX)")
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
        briefing, scores = run_and_save()
    briefing_date = date.today().isoformat()
else:
    briefing, briefing_date, scores = load_latest_briefing()

last_updated_slot.markdown(f"**Last updated:** {briefing_date or '—'}")

if not briefing:
    st.info("No briefing found. Click **↻ Refresh** to generate one.")
    st.stop()

# ── price strip ───────────────────────────────────────────────────────────────

col_si, col_gc, col_ratio, col_macro = st.columns(4)
with col_si:
    st.markdown(
        price_widget_html("Silver (SI=F)", f"${silver['price']:.2f}", silver["change"], silver["change_pct"]),
        unsafe_allow_html=True,
    )
with col_gc:
    st.markdown(
        price_widget_html("Gold (GC=F)", f"${gold['price']:,.2f}", gold["change"], gold["change_pct"]),
        unsafe_allow_html=True,
    )
with col_ratio:
    st.markdown(
        '<div style="font-size:0.65rem;color:#5a6a7e;font-weight:600;text-transform:uppercase;'
        'letter-spacing:0.09em;margin-bottom:5px;">Gold / Silver Ratio</div>'
        f'<div style="font-size:1.9rem;font-weight:700;color:#d4af37;'
        f'font-family:\'SF Mono\',monospace;line-height:1.1;">{ratio:.1f}</div>'
        '<div style="font-size:0.78rem;color:#2d3f5a;margin-top:3px;">Hist. avg ~65</div>',
        unsafe_allow_html=True,
    )
with col_macro:
    st.markdown(
        macro_widget_html([
            ("DXY (Dollar Index)", f"{dxy['price']:.2f}", dxy["change"], dxy["change_pct"]),
            ("US 10Y Yield", f"{us10y['price']:.2f}%", us10y["change"], us10y["change_pct"]),
        ]),
        unsafe_allow_html=True,
    )

st.divider()

# ── significant move banner ───────────────────────────────────────────────────

_sig_pct = silver["change_pct"]
if abs(_sig_pct) >= 2.0:
    _sign = "+" if _sig_pct > 0 else ""
    if _sig_pct > 0:
        _bg, _border, _fg = "#061a0e", "#145230", "#38a169"
    else:
        _bg, _border, _fg = "#1a0606", "#7a1c1c", "#e53e3e"
    st.markdown(
        f'<div style="background:{_bg};border:1px solid {_border};border-radius:4px;'
        f'padding:7px 16px;margin-bottom:6px;font-size:0.84rem;font-weight:600;color:{_fg};">'
        f'⚡ Significant move detected: {_sign}{_sig_pct:.2f}%'
        f' — briefing focused on move drivers'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── supply risk bar ───────────────────────────────────────────────────────────

supply_level = scores.get("supply_risk", "LOW")
supply_reason = extract_supply_risk_reason(briefing)
st.markdown(
    '<div style="display:flex;align-items:center;gap:14px;padding:6px 0;">'
    '<div style="font-size:0.65rem;color:#5a6a7e;font-weight:600;text-transform:uppercase;'
    'letter-spacing:0.09em;white-space:nowrap;">Supply Risk</div>'
    f'{supply_risk_badge_html(supply_level)}'
    f'<div style="font-size:0.8rem;color:#5a6a7e;">{escape_dollars(supply_reason)}</div>'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()

# ── main 2:1 layout ───────────────────────────────────────────────────────────

left_col, right_col = st.columns([2, 1])

with left_col:
    history = cached_silver_history()
    if history:
        st.plotly_chart(render_silver_chart(history), width="stretch", config={"displayModeBar": False})

    st.markdown(
        '<div style="font-size:0.65rem;color:#5a6a7e;font-weight:600;text-transform:uppercase;'
        'letter-spacing:0.09em;margin-bottom:8px;">Top Stories by Impact</div>',
        unsafe_allow_html=True,
    )
    top_stories = extract_top_stories(briefing)
    st.markdown(escape_dollars(top_stories) if top_stories else "_No top stories extracted._")

    with st.expander("Full market analysis"):
        st.markdown(escape_dollars(briefing))

with right_col:
    overall = int(scores.get("overall", 5))
    st.markdown(
        '<div style="text-align:center;padding:16px 0 8px;">'
        '<div style="font-size:0.65rem;color:#5a6a7e;font-weight:600;text-transform:uppercase;'
        'letter-spacing:0.09em;margin-bottom:6px;">Overall Conviction</div>'
        f'<div style="font-size:3.5rem;font-weight:800;color:{score_color(overall)};'
        f'font-family:\'SF Mono\',monospace;line-height:1;">{overall}</div>'
        '<div style="font-size:0.7rem;color:#2d3f5a;margin-top:2px;">/ 10</div>'
        f'<div style="margin-top:10px;">{overall_badge_html(overall)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    verdict = scores.get("verdict", "")
    if verdict:
        st.markdown(
            f'<div style="font-size:0.82rem;color:#8a9ab5;font-style:italic;text-align:center;'
            f'padding:8px 12px;border-top:1px solid #1a2540;border-bottom:1px solid #1a2540;'
            f'margin:8px 0;">{escape_dollars(verdict)}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown(
        '<div style="font-size:0.65rem;color:#5a6a7e;font-weight:600;text-transform:uppercase;'
        'letter-spacing:0.09em;margin-bottom:10px;">Conviction Score</div>',
        unsafe_allow_html=True,
    )
    _components = [
        ("Macro", "macro"),
        ("Technicals", "technicals"),
        ("Sentiment", "sentiment"),
        ("ETF Flows", "etf_flows"),
        ("Industrial Demand", "industrial_demand"),
    ]
    for label, key in _components:
        val = int(scores.get(key, 5))
        st.markdown(colored_label_html(f"{label}  {val}/10", val), unsafe_allow_html=True)
        st.progress(val / 10)

    st.divider()

    bullish, bearish = extract_bull_bear(briefing)
    if bullish or bearish:
        st.markdown(
            '<div style="font-size:0.65rem;color:#5a6a7e;font-weight:600;text-transform:uppercase;'
            'letter-spacing:0.09em;margin-bottom:8px;">Bull vs Bear</div>',
            unsafe_allow_html=True,
        )
        b_col, br_col = st.columns(2)
        with b_col:
            st.markdown('<div style="font-size:0.7rem;color:#38a169;font-weight:700;margin-bottom:4px;">BULLISH</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:0.78rem;color:#6a7a8e;">{escape_dollars(bullish)}</div>', unsafe_allow_html=True)
        with br_col:
            st.markdown('<div style="font-size:0.7rem;color:#e53e3e;font-weight:700;margin-bottom:4px;">BEARISH</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:0.78rem;color:#6a7a8e;">{escape_dollars(bearish)}</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown(
        '<div style="font-size:0.65rem;color:#5a6a7e;font-weight:600;text-transform:uppercase;'
        'letter-spacing:0.09em;margin-bottom:6px;">Ask the Silver Analyst</div>',
        unsafe_allow_html=True,
    )
    with st.container(height=160):
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    with st.form("analyst_form", clear_on_submit=True):
        user_question = st.text_input(
            "Question",
            placeholder="Ask me anything about silver…",
            label_visibility="collapsed",
        )
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

# ── footer ────────────────────────────────────────────────────────────────────

st.divider()
foot_left, foot_right = st.columns([3, 1])
with foot_left:
    st.markdown(
        f'<div style="font-size:0.72rem;color:#2d3f5a;">'
        f'Briefing date: {briefing_date or "—"}&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'Data: Yahoo Finance (SI=F, GC=F, DXY, US10Y) &amp; Google News RSS'
        f'</div>',
        unsafe_allow_html=True,
    )
with foot_right:
    pdf_bytes = generate_pdf(briefing, silver, gold, briefing_date or "")
    st.download_button(
        "⬇ PDF",
        data=pdf_bytes,
        file_name=f"silver_briefing_{briefing_date}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
