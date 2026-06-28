import io
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv
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


# ── disk helpers ──────────────────────────────────────────────────────────────

def load_latest_briefing() -> tuple[str | None, str | None, dict]:
    files = sorted(Path(OUTPUTS_DIR).glob("briefing_*.txt"))
    if not files:
        return None, None, {}
    f = files[-1]
    date_str = f.stem.replace("briefing_", "")
    raw = f.read_text()
    briefing_text, scores = extract_scores(raw)
    scores_file = Path(OUTPUTS_DIR) / f"scores_{date_str}.json"
    if scores_file.exists():
        try:
            scores = json.loads(scores_file.read_text())
        except Exception:
            pass
    return briefing_text, date_str, scores


def load_latest_prices() -> tuple[dict, dict, dict, dict] | None:
    files = sorted(Path(OUTPUTS_DIR).glob("prices_*.json"))
    if not files:
        return None
    try:
        data = json.loads(files[-1].read_text())
        return data["silver"], data["gold"], data["dxy"], data["us10y"]
    except Exception:
        return None


def load_latest_history() -> list[dict]:
    files = sorted(Path(OUTPUTS_DIR).glob("history_*.json"))
    if not files:
        return []
    try:
        return json.loads(files[-1].read_text())
    except Exception:
        return []


def run_and_save() -> tuple[str, dict, dict, dict, dict, dict, list[dict]]:
    silver, gold, dxy, us10y = (
        fetch_silver_price(), fetch_gold_price(), fetch_dxy_price(), fetch_us10y_price()
    )
    history = fetch_silver_history(30)
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
    (out_dir / f"prices_{today}.json").write_text(
        json.dumps({"silver": silver, "gold": gold, "dxy": dxy, "us10y": us10y})
    )
    (out_dir / f"history_{today}.json").write_text(json.dumps(history))
    return briefing_text, scores, silver, gold, dxy, us10y, history


# ── text helpers ──────────────────────────────────────────────────────────────

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


def extract_top_stories(text: str) -> str:
    m = re.search(r"TOP STORIES BY IMPACT[^\n]*\n(.*?)(?=\n#{1,3}\s+[A-Z]|\n\*\*[A-Z]|\n[A-Z][A-Z\s\/]{7,}|\Z)", text, re.DOTALL)
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


# ── PDF ───────────────────────────────────────────────────────────────────────

def generate_pdf(briefing: str, silver: dict, gold: dict, briefing_date: str) -> bytes:
    # Strip all markdown links before rendering so no raw URLs leak into the PDF
    briefing = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', briefing)

    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import HRFlowable, SimpleDocTemplate, Paragraph, Spacer

    NAVY = HexColor("#1a2f5e")
    GREEN = HexColor("#1a5e3a")
    BODY = HexColor("#2c2c2c")
    GRAY = HexColor("#888888")

    margin = 2 * cm
    page_w, _ = A4

    def _s(name, **kw):
        base = dict(fontName="Helvetica", fontSize=10, leading=14, textColor=BODY)
        base.update(kw)
        return ParagraphStyle(name, **base)

    s_title   = _s("title",    fontName="Helvetica-Bold",    fontSize=20, textColor=NAVY,  leading=24, spaceAfter=3)
    s_sub     = _s("sub",      fontSize=9,                   textColor=GRAY,  leading=12, spaceAfter=2)
    s_prices  = _s("prices",   fontName="Helvetica-Bold",    fontSize=9,  textColor=BODY,  leading=13, spaceAfter=0)
    s_section = _s("section",  fontName="Helvetica-Bold",    fontSize=13, textColor=GREEN, leading=17, spaceBefore=10, spaceAfter=2)
    s_body    = _s("body",     fontSize=10, textColor=BODY,  leading=14, spaceAfter=2)
    s_bullet  = _s("bullet",   fontSize=10, textColor=BODY,  leading=14, leftIndent=14, spaceAfter=2)
    s_lbl     = _s("lbl",      fontName="Helvetica-Bold",    fontSize=10, textColor=NAVY, leading=14)
    s_rsn     = _s("rsn",      fontName="Helvetica-Oblique", fontSize=9,  textColor=GRAY, leading=13, leftIndent=14, spaceAfter=5)
    s_story_h = _s("story_h",  fontName="Helvetica-Bold",    fontSize=10, textColor=NAVY, leading=14)
    s_story_r = _s("story_r",  fontName="Helvetica-Oblique", fontSize=9,  textColor=GRAY, leading=13, leftIndent=14, spaceAfter=6)

    def md_to_rl(text: str) -> str:
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"\*(.+?)\*",     r"<i>\1</i>", text)
        text = re.sub(r"[*_`]", "", text)
        return text

    def hr_navy():
        return HRFlowable(width="100%", thickness=1.5, color=NAVY,  spaceAfter=8, spaceBefore=4)

    def hr_green():
        return HRFlowable(width="100%", thickness=0.5, color=GREEN, spaceAfter=6, spaceBefore=1)

    def draw_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(GRAY)
        canvas.drawString(margin, margin * 0.55, "Silver Market Intelligence — Confidential")
        canvas.drawRightString(page_w - margin, margin * 0.55, f"Page {doc.page}")
        canvas.restoreState()

    def parse_score(line: str):
        m = re.match(
            r"(Macro|Technicals|Sentiment|Supply\s*Risk|Overall)"
            r"[:\s]+(\d+/10|LOW|MEDIUM|HIGH)[^—\-]*[—\-]+\s*(.+)",
            line, re.IGNORECASE,
        )
        return (f"{m.group(1).upper()}: {m.group(2)}", m.group(3).strip()) if m else None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=margin, bottomMargin=margin * 1.6,
        leftMargin=margin, rightMargin=margin,
    )
    elems = []

    # ── Title block ───────────────────────────────────────────────────────────
    elems.append(Paragraph("Silver Market Intelligence", s_title))
    elems.append(Paragraph(f"Daily Briefing — {briefing_date}", s_sub))
    elems.append(hr_navy())

    # ── Price strip ───────────────────────────────────────────────────────────
    ratio = gold["price"] / silver["price"]
    sign_s = "+" if silver["change"] >= 0 else ""
    sign_g = "+" if gold["change"] >= 0 else ""
    elems.append(Paragraph(
        f"<b>Silver:</b> ${silver['price']:.2f} ({sign_s}{silver['change_pct']:.2f}%)"
        f"&nbsp;&nbsp;&nbsp;<b>Gold:</b> ${gold['price']:,.2f} ({sign_g}{gold['change_pct']:.2f}%)"
        f"&nbsp;&nbsp;&nbsp;<b>Au/Ag Ratio:</b> {ratio:.1f}",
        s_prices,
    ))
    elems.append(Spacer(1, 0.35 * cm))

    # ── Line-by-line briefing render ──────────────────────────────────────────
    for line in briefing.split("\n"):
        stripped = line.strip()

        if not stripped or stripped == "---":
            elems.append(Spacer(1, 3))
            continue

        # Box-drawing / repeated-dash dividers (━━━, ----, ════ etc.) → green HR
        if re.match(r'^[━─■=\-]{4,}$', stripped):
            elems.append(HRFlowable(width="100%", thickness=0.5, color=GREEN, spaceAfter=6, spaceBefore=6))
            continue

        # Markdown link story: "1. [Title](URL) — reason"
        sm = re.match(r"(\d+)\.\s+\[([^\]]+)\]\(([^)]+)\)\s*[—–\-]+\s*(.+)", stripped)
        if sm:
            num, title, url, reason = sm.group(1), sm.group(2), sm.group(3), sm.group(4)
            safe_url   = url.replace("&", "&amp;")
            safe_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            elems.append(Paragraph(
                f'{num}.&nbsp;&nbsp;<a href="{safe_url}"><font color="#1a2f5e"><b>{safe_title}</b></font></a>',
                s_story_h,
            ))
            elems.append(Paragraph(md_to_rl(reason), s_story_r))
            continue

        # Markdown headings
        if stripped.startswith("### "):
            elems.append(Paragraph(md_to_rl(stripped[4:]), s_section))
            elems.append(hr_green())
            continue
        if stripped.startswith("## ") or stripped.startswith("# "):
            elems.append(Paragraph(md_to_rl(stripped.lstrip("#").strip()), s_section))
            elems.append(hr_green())
            continue

        # ALL-CAPS section header (briefing convention)
        if re.match(r"^[A-Z][A-Z\s\/]+$", stripped) and len(stripped) >= 8:
            elems.append(Spacer(1, 4))
            elems.append(Paragraph(stripped, s_section))
            elems.append(hr_green())
            continue

        # Score lines: "Macro: 7/10 — explanation"
        sc = parse_score(stripped)
        if sc:
            elems.append(Paragraph(sc[0], s_lbl))
            elems.append(Paragraph(md_to_rl(sc[1]), s_rsn))
            continue

        # Bullet points
        if re.match(r"^[-*]\s", stripped):
            cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", stripped[2:])
            elems.append(Paragraph(f"• {md_to_rl(cleaned)}", s_bullet))
            continue

        # Body text — strip any remaining markdown links
        cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", stripped)
        elems.append(Paragraph(md_to_rl(cleaned), s_body))

    doc.build(elems, onFirstPage=draw_footer, onLaterPages=draw_footer)
    return buffer.getvalue()


# ── analyst chat ──────────────────────────────────────────────────────────────

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


# ── Bloomberg-style helpers ───────────────────────────────────────────────────

def svg_sparkline(closes: list[float], width: int = 80, height: int = 30) -> str:
    if not closes or len(closes) < 2:
        return ""
    mn, mx = min(closes), max(closes)
    rng = mx - mn or 1
    pts = []
    for i, v in enumerate(closes):
        x = round(i / (len(closes) - 1) * width, 1)
        y = round(height - (v - mn) / rng * height, 1)
        pts.append(f"{x},{y}")
    color = "#00d4aa" if closes[-1] >= closes[0] else "#ff4757"
    path = "M " + " L ".join(pts)
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.5" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'</svg>'
    )


def parse_story_list(briefing: str) -> list[dict]:
    section = extract_top_stories(briefing) or briefing
    stories = []
    for m in re.finditer(r"\d+\.\s+\[([^\]]+)\]\(([^)]+)\)\s*[—–\-]+\s*(.+)", section):
        title, url, reason = m.group(1), m.group(2), m.group(3).strip()
        combined = (title + " " + reason).lower()
        if any(k in combined for k in ("fund", "etf", "trust", "holding")):
            category, cat_color = "Funds/ETF", "#00d4aa"
        elif any(k in combined for k in ("fed", "dollar", "inflation", "rate", "gdp", "tariff", "macro", "economic")):
            category, cat_color = "Macro", "#ff4757"
        elif any(k in combined for k in ("demand", "industrial", "solar", "electric", "manufactur")):
            category, cat_color = "Market", "#ffa500"
        else:
            category, cat_color = "Analysts", "#f59e0b"
        stories.append({"title": title, "url": url, "reason": reason, "category": category, "cat_color": cat_color})
    return stories[:5]


def extract_component_reasons(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, pat in [
        ("macro", r"Macro:\s*\d+/10\s*[—–\-]+\s*(.+)"),
        ("technicals", r"Technicals:\s*\d+/10\s*[—–\-]+\s*(.+)"),
        ("sentiment", r"Sentiment:\s*\d+/10\s*[—–\-]+\s*(.+)"),
        ("supply_risk", r"Risk:\s*(?:LOW|MEDIUM|HIGH)[^\n]*\n([^\n]+)"),
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        result[key] = m.group(1).strip() if m else ""
    return result


def score_to_status(score: int | float) -> tuple[str, str]:
    if score <= 4:
        return "BEARISH", "#ff4757"
    if score <= 6:
        return "NEUTRAL", "#ffa500"
    return "BULLISH", "#00d4aa"


def supply_status(level: str) -> tuple[str, str]:
    return {"HIGH": ("HIGH", "#ff4757"), "MEDIUM": ("MEDIUM", "#ffa500")}.get(
        level.upper(), ("LOW", "#00d4aa")
    )


# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
<style>
/* base reset */
.stApp, .main, .main .block-container {
    background-color: #0a0e1a !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    max-width: 100% !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}
.block-container { padding: 1rem 1.5rem 1rem 1.5rem !important; }

/* sidebar */
section[data-testid="stSidebar"] > div {
    background-color: #0d1117 !important;
    border-right: 1px solid #1a2035 !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] .stMarkdown p {
    color: #4a5a72 !important;
    font-size: 0.8rem !important;
}

/* typography */
h1, h2, h3, h4 { color: #ffffff !important; }
p, li, .stMarkdown p { color: #8a9ab5 !important; }
label { color: #5a6a7e !important; }

/* dividers */
hr { border-color: #1a2035 !important; margin: 0.3rem 0 !important; }

/* inputs */
[data-testid="stTextInput"] input {
    background-color: #0d1117 !important;
    color: #c8d4e8 !important;
    border: 1px solid #1a2035 !important;
    border-radius: 3px !important;
    font-size: 0.82rem !important;
}
[data-testid="stTextInput"] input::placeholder { color: #2d3f5a !important; }

/* primary button */
button[kind="primary"] {
    background-color: #00d4aa !important;
    color: #0a0e1a !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    border-radius: 3px !important;
}
button[kind="primary"]:hover { background-color: #00e6bb !important; }

/* secondary button */
button[kind="secondary"] {
    background-color: transparent !important;
    color: #2d3f5a !important;
    border: 1px solid #1a2035 !important;
    font-size: 0.7rem !important;
    border-radius: 3px !important;
}

/* download button */
[data-testid="stDownloadButton"] button {
    background-color: #132040 !important;
    color: #7aa8e0 !important;
    border: 1px solid #1e3460 !important;
    font-weight: 600 !important;
    border-radius: 3px !important;
    font-size: 0.82rem !important;
}

/* form submit */
[data-testid="stFormSubmitButton"] button {
    background-color: #0d1117 !important;
    color: #00d4aa !important;
    border: 1px solid #00d4aa !important;
    border-radius: 3px !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
}
[data-testid="stFormSubmitButton"] button:hover {
    background-color: #00d4aa !important;
    color: #0a0e1a !important;
}

/* chat */
[data-testid="stChatMessage"] {
    background-color: #0d1117 !important;
    border-radius: 3px !important;
}

/* border wrapper */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #1a2035 !important;
    border-radius: 4px !important;
}

/* caption */
[data-testid="stCaptionContainer"] { color: #2d3f5a !important; }

/* hide plotly modebar */
.js-plotly-plot .modebar { display: none !important; }
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

# ── sidebar ───────────────────────────────────────────────────────────────────

_NAV_ITEMS = [
    ("Overview", True),
    ("News & Events", False),
    ("Drivers", False),
    ("Macro", False),
    ("Supply Risk", False),
    ("Reports", False),
    ("Alerts", False),
]

with st.sidebar:
    st.markdown(
        '<div style="padding:20px 16px 14px;border-bottom:1px solid #1a2035;">'
        '<div style="font-size:1.05rem;font-weight:800;color:#ffffff;letter-spacing:-0.01em;">'
        '🪙 Silver Market Intelligence</div>'
        '<div style="font-size:0.7rem;color:#4a5a72;margin-top:3px;">AI-powered metals briefing</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="padding:8px 0;">', unsafe_allow_html=True)
    for label, active in _NAV_ITEMS:
        if active:
            st.markdown(
                f'<div style="padding:9px 16px;background:#1a2035;border-left:3px solid #00d4aa;'
                f'font-size:0.82rem;font-weight:700;color:#ffffff;margin:1px 0;">{label}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="padding:9px 16px;font-size:0.82rem;color:#8a9ab5;'
                f'opacity:0.4;margin:1px 0;">{label}</div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        '<div style="padding:9px 16px;font-size:0.82rem;color:#8a9ab5;opacity:0.4;margin:1px 0;'
        'display:flex;align-items:center;gap:7px;">Ask the Analyst'
        '<span style="background:#2d1a4a;color:#c084fc;font-size:0.6rem;font-weight:700;'
        'padding:1px 6px;border-radius:3px;letter-spacing:0.05em;">BETA</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div style="padding:12px 16px;border-top:1px solid #1a2035;">', unsafe_allow_html=True)
    last_updated_slot = st.empty()
    st.markdown(
        '<div style="display:flex;align-items:center;gap:6px;margin-top:6px;">'
        '<div style="width:7px;height:7px;border-radius:50%;background:#00d4aa;'
        'box-shadow:0 0 4px #00d4aa;"></div>'
        '<div style="font-size:0.7rem;color:#4a5a72;">Auto-refresh in 60s</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ── header row ────────────────────────────────────────────────────────────────

_now = datetime.now()
_date_label = _now.strftime(f"%A, %B {_now.day}, %Y")
st.markdown(
    '<h1 style="color:#ffffff;font-size:1.6rem;font-weight:800;margin:2px 0 2px;'
    'letter-spacing:-0.02em;line-height:1.2;">Silver Market Overview</h1>'
    f'<div style="font-size:0.82rem;color:#4a5a72;margin-bottom:6px;">{_date_label}</div>',
    unsafe_allow_html=True,
)

st.divider()

# ── load / generate briefing ──────────────────────────────────────────────────

briefing, briefing_date, scores = load_latest_briefing()
_prices = load_latest_prices()
if _prices is not None:
    silver, gold, dxy, us10y = _prices
else:
    silver = gold = dxy = us10y = None
history = load_latest_history()

last_updated_slot.markdown(
    f'<div style="font-size:0.7rem;color:#4a5a72;">Last updated<br>'
    f'<span style="color:#8a9ab5;">{briefing_date or "—"}</span></div>',
    unsafe_allow_html=True,
)

if not briefing:
    st.info("No briefing available. Run the backend to generate one.")
    st.stop()

# ── price strip ───────────────────────────────────────────────────────────────

if silver is not None:
    ratio = gold["price"] / silver["price"]
    _spark_closes = [h["close"] for h in history] if history else []
    _spark = svg_sparkline(_spark_closes)

    def _price_card(label: str, price_str: str, change: float, pct: float, spark: str = "") -> str:
        color = "#00d4aa" if change >= 0 else "#ff4757"
        sign = "+" if change >= 0 else ""
        spark_div = f'<div style="position:absolute;top:10px;right:10px;opacity:0.8;">{spark}</div>' if spark else ""
        return (
            f'<div style="background:#0d1117;border:1px solid #1a2035;border-radius:6px;'
            f'padding:14px 14px 10px;position:relative;min-height:86px;">'
            f'{spark_div}'
            f'<div style="font-size:0.62rem;color:#4a5a72;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:0.09em;margin-bottom:4px;">{label}</div>'
            f'<div style="font-size:1.7rem;font-weight:700;color:#ffffff;'
            f'font-family:\'SF Mono\',\'Fira Mono\',monospace;line-height:1.1;">{price_str}</div>'
            f'<div style="font-size:0.78rem;color:{color};font-weight:600;margin-top:3px;">'
            f'{sign}{change:.2f}&nbsp;({sign}{pct:.2f}%)</div>'
            f'</div>'
        )

    def _ratio_card(r: float) -> str:
        return (
            '<div style="background:#0d1117;border:1px solid #1a2035;border-radius:6px;'
            'padding:14px 14px 10px;min-height:86px;">'
            '<div style="font-size:0.62rem;color:#4a5a72;font-weight:600;text-transform:uppercase;'
            'letter-spacing:0.09em;margin-bottom:4px;">Gold / Silver Ratio</div>'
            f'<div style="font-size:1.7rem;font-weight:700;color:#d4af37;'
            f'font-family:\'SF Mono\',monospace;line-height:1.1;">{r:.1f}</div>'
            '<div style="font-size:0.72rem;color:#4a5a72;margin-top:3px;">Hist. avg ~65</div>'
            '</div>'
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(_price_card("Silver (SI=F)", f"${silver['price']:.2f}", silver["change"], silver["change_pct"], _spark), unsafe_allow_html=True)
    with c2:
        st.markdown(_price_card("Gold (GC=F)", f"${gold['price']:,.2f}", gold["change"], gold["change_pct"], _spark), unsafe_allow_html=True)
    with c3:
        st.markdown(_ratio_card(ratio), unsafe_allow_html=True)
    with c4:
        st.markdown(_price_card("DXY (Dollar Index)", f"{dxy['price']:.2f}", dxy["change"], dxy["change_pct"]), unsafe_allow_html=True)
    with c5:
        st.markdown(_price_card("US 10Y Yield", f"{us10y['price']:.2f}%", us10y["change"], us10y["change_pct"]), unsafe_allow_html=True)

    # ── significant move banner ───────────────────────────────────────────────

    _sig_pct = silver["change_pct"]
    if abs(_sig_pct) >= 2.0:
        _sign = "+" if _sig_pct > 0 else ""
        _fg = "#00d4aa" if _sig_pct > 0 else "#ff4757"
        _border = "#1a4a2a" if _sig_pct > 0 else "#4a1a1a"
        st.markdown(
            '<div style="margin-top:12px;">'
            f'<div style="background:#0d2818;border:1px solid {_border};border-radius:4px;'
            f'padding:10px 18px;display:flex;align-items:center;justify-content:space-between;">'
            f'<div>'
            f'<span style="color:#ffa500;margin-right:8px;font-size:1rem;">⚡</span>'
            f'<span style="font-size:0.82rem;font-weight:700;color:{_fg};text-transform:uppercase;'
            f'letter-spacing:0.06em;">SIGNIFICANT MOVE DETECTED</span>'
            f'<span style="font-size:0.82rem;color:#c8d4e8;margin-left:10px;">'
            f'Silver {_sign}{_sig_pct:.2f}% — briefing focused on move drivers</span>'
            f'</div>'
            f'<span style="font-size:0.78rem;color:{_fg};white-space:nowrap;cursor:pointer;">'
            f'View details →</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown('<div style="margin-top:14px;"></div>', unsafe_allow_html=True)

# ── main 60/40 layout ─────────────────────────────────────────────────────────

left_col, right_col = st.columns([3, 2])

with left_col:
    # ── top news card ─────────────────────────────────────────────────────────
    stories = parse_story_list(briefing)

    news_rows = []
    if stories:
        for i, s in enumerate(stories):
            impact_color = "#ff4757" if i < 2 else "#ffa500"
            impact_label = "High" if i < 2 else "Medium"
            border = "border-top:1px solid #1a2035;" if i > 0 else ""
            news_rows.append(
                f'<div style="display:flex;align-items:flex-start;gap:12px;padding:10px 0;{border}">'
                f'<div style="font-size:1.4rem;font-weight:800;color:#1a2540;min-width:22px;'
                f'line-height:1;margin-top:2px;">{i + 1}</div>'
                f'<div style="width:9px;height:9px;border-radius:50%;background:{s["cat_color"]};'
                f'flex-shrink:0;margin-top:5px;"></div>'
                f'<div style="flex:1;min-width:0;">'
                f'<div style="font-size:0.82rem;color:#c8d4e8;font-weight:600;line-height:1.35;'
                f'margin-bottom:3px;">'
                f'<a href="{s["url"]}" target="_blank" style="color:inherit;text-decoration:none;">'
                f'{s["title"]}</a></div>'
                f'<div style="font-size:0.72rem;color:#4a5a72;margin-bottom:5px;line-height:1.4;">'
                f'{s["reason"][:120]}{"…" if len(s["reason"]) > 120 else ""}</div>'
                f'<div style="display:flex;align-items:center;gap:8px;">'
                f'<span style="background:#1a2035;color:{s["cat_color"]};font-size:0.6rem;'
                f'font-weight:700;padding:1px 7px;border-radius:3px;letter-spacing:0.04em;">'
                f'{s["category"]}</span>'
                f'<span style="font-size:0.67rem;color:#2d3f5a;">Today</span>'
                f'<span style="font-size:0.67rem;color:{impact_color};font-weight:700;">{impact_label}</span>'
                f'</div></div></div>'
            )
    else:
        top_raw = extract_top_stories(briefing)
        news_rows.append(
            f'<div style="font-size:0.82rem;color:#8a9ab5;padding:8px 0;">'
            f'{top_raw if top_raw else "No top stories extracted."}'
            f'</div>'
        )

    st.markdown(
        '<div style="background:#0d1117;border:1px solid #1a2035;border-radius:6px;'
        'padding:18px 20px;margin-bottom:14px;">'
        '<div style="display:flex;align-items:center;gap:6px;margin-bottom:12px;">'
        '<div style="font-size:0.65rem;font-weight:700;color:#4a5a72;text-transform:uppercase;'
        'letter-spacing:0.09em;">Top News by Impact</div>'
        '<span style="font-size:0.7rem;color:#2d3f5a;cursor:help;" '
        'title="Top 5 most market-impactful stories from today\'s news feed">(?)</span>'
        '</div>'
        + "".join(news_rows) +
        '<div style="padding-top:10px;border-top:1px solid #1a2035;margin-top:4px;">'
        '<span style="font-size:0.75rem;color:#4a5a72;cursor:pointer;">View all news →</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── AI morning brief card ─────────────────────────────────────────────────
    verdict = scores.get("verdict", "")
    st.markdown(
        '<div style="background:#0d1117;border:1px solid #1a2035;border-radius:6px;'
        'padding:18px 20px;">'
        '<div style="display:flex;align-items:center;gap:6px;margin-bottom:12px;">'
        '<div style="font-size:0.65rem;font-weight:700;color:#4a5a72;text-transform:uppercase;'
        'letter-spacing:0.09em;">AI Morning Brief</div>'
        '<span style="font-size:0.7rem;color:#2d3f5a;cursor:help;" '
        'title="Generated by Claude AI from today\'s market data and news">(?)</span>'
        '</div>'
        f'<div style="font-size:0.85rem;color:#8a9ab5;line-height:1.65;">'
        f'{verdict if verdict else "No briefing summary available."}'
        f'</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    _exp_col, _pdf_col = st.columns([4, 1])
    with _exp_col:
        with st.expander("View full analysis"):
            st.markdown(briefing)
    with _pdf_col:
        if silver is not None:
            pdf_bytes = generate_pdf(briefing, silver, gold, briefing_date or "")
            st.download_button(
                "⬇ PDF",
                data=pdf_bytes,
                file_name=f"silver_briefing_{briefing_date}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

with right_col:
    # ── market snapshot card ──────────────────────────────────────────────────
    reasons = extract_component_reasons(briefing)
    _snap_rows = [
        ("Macro", score_to_status(int(scores.get("macro", 5))), reasons.get("macro", "")),
        ("Technicals", score_to_status(int(scores.get("technicals", 5))), reasons.get("technicals", "")),
        ("Supply Risk", supply_status(scores.get("supply_risk", "LOW")), reasons.get("supply_risk", extract_supply_risk_reason(briefing))),
        ("Sentiment", score_to_status(int(scores.get("sentiment", 5))), reasons.get("sentiment", "")),
    ]

    snap_row_html = ""
    for i, (label, (status, status_color), reason) in enumerate(_snap_rows):
        border = "border-top:1px solid #1a2035;" if i > 0 else ""
        short_reason = reason[:88] + ("…" if len(reason) > 88 else "")
        snap_row_html += (
            f'<div style="display:flex;align-items:flex-start;gap:10px;padding:9px 0;{border}">'
            f'<div style="width:9px;height:9px;border-radius:50%;background:{status_color};'
            f'flex-shrink:0;margin-top:3px;"></div>'
            f'<div style="flex:1;min-width:0;">'
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:2px;">'
            f'<span style="font-size:0.67rem;color:#4a5a72;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:0.07em;">{label}</span>'
            f'<span style="font-size:0.67rem;color:{status_color};font-weight:700;'
            f'letter-spacing:0.06em;">{status}</span>'
            f'</div>'
            f'<div style="font-size:0.72rem;color:#4a5a72;line-height:1.4;">{short_reason}</div>'
            f'</div></div>'
        )

    overall = int(scores.get("overall", 5))
    o_status, o_color = score_to_status(overall)

    st.markdown(
        '<div style="background:#0d1117;border:1px solid #1a2035;border-radius:6px;'
        'padding:18px 20px;margin-bottom:14px;">'
        '<div style="display:flex;align-items:center;gap:6px;margin-bottom:12px;">'
        '<div style="font-size:0.65rem;font-weight:700;color:#4a5a72;text-transform:uppercase;'
        'letter-spacing:0.09em;">Market Snapshot</div>'
        '<span style="font-size:0.7rem;color:#2d3f5a;cursor:help;" '
        'title="Current stance for each market driver">(?)</span>'
        '</div>'
        + snap_row_html +
        f'<div style="display:flex;align-items:center;gap:14px;padding:12px 0 0;'
        f'border-top:1px solid #1a2035;margin-top:4px;">'
        f'<div style="font-size:0.67rem;color:#4a5a72;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.07em;">Overall Conviction</div>'
        f'<div style="border:1px solid {o_color};border-radius:4px;padding:2px 10px;'
        f'font-size:1.1rem;font-weight:800;color:{o_color};font-family:\'SF Mono\',monospace;">'
        f'{overall}/10</div>'
        f'<span style="background:{o_color}22;color:{o_color};font-size:0.67rem;font-weight:700;'
        f'padding:2px 10px;border-radius:3px;letter-spacing:0.07em;">{o_status}</span>'
        f'</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── watchlist card ────────────────────────────────────────────────────────
    _events = [
        ("14:30", "US Core PCE Price Index (MoM)", "High"),
        ("14:30", "US Personal Income (MoM)", "Medium"),
        ("16:00", "Fed Chair Powell Speech", "High"),
        ("22:30", "China Manufacturing PMI", "Medium"),
        ("All Day", "OPEC+ Meeting", "Medium"),
    ]

    def _dots(level: str) -> str:
        if level == "High":
            return '<span style="color:#ff4757;letter-spacing:1px;">●●●</span>'
        return '<span style="color:#ffa500;letter-spacing:1px;">●●</span><span style="color:#2d3f5a;letter-spacing:1px;">●</span>'

    event_rows_html = ""
    for i, (t, evt, imp) in enumerate(_events):
        border = "border-top:1px solid #0f1525;" if i > 0 else ""
        event_rows_html += (
            f'<div style="display:flex;align-items:center;padding:7px 0;{border}">'
            f'<div style="font-size:0.72rem;color:#4a5a72;font-family:\'SF Mono\',monospace;width:58px;flex-shrink:0;">{t}</div>'
            f'<div style="font-size:0.75rem;color:#c8d4e8;flex:1;">{evt}</div>'
            f'<div style="width:46px;text-align:center;font-size:0.65rem;">{_dots(imp)}</div>'
            f'</div>'
        )

    st.markdown(
        '<div style="background:#0d1117;border:1px solid #1a2035;border-radius:6px;padding:18px 20px;">'
        '<div style="display:flex;align-items:center;gap:6px;margin-bottom:12px;">'
        '<div style="font-size:0.65rem;font-weight:700;color:#4a5a72;text-transform:uppercase;'
        'letter-spacing:0.09em;">Today\'s Watchlist</div>'
        '<span style="font-size:0.7rem;color:#2d3f5a;cursor:help;" '
        'title="Key scheduled events for today">(?)</span>'
        '</div>'
        '<div style="display:flex;padding-bottom:6px;border-bottom:1px solid #1a2035;margin-bottom:2px;">'
        '<div style="font-size:0.6rem;color:#2d3f5a;font-weight:600;text-transform:uppercase;'
        'letter-spacing:0.07em;width:58px;">TIME (UTC)</div>'
        '<div style="font-size:0.6rem;color:#2d3f5a;font-weight:600;text-transform:uppercase;'
        'letter-spacing:0.07em;flex:1;">EVENT</div>'
        '<div style="font-size:0.6rem;color:#2d3f5a;font-weight:600;text-transform:uppercase;'
        'letter-spacing:0.07em;width:46px;text-align:center;">IMPACT</div>'
        '</div>'
        + event_rows_html +
        '<div style="padding-top:10px;border-top:1px solid #1a2035;margin-top:4px;">'
        '<span style="font-size:0.75rem;color:#4a5a72;cursor:pointer;">View full calendar →</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

# ── footer ────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    f'<div style="font-size:0.72rem;color:#2d3f5a;">'
    f'Briefing date: {briefing_date or "—"}&nbsp;&nbsp;|&nbsp;&nbsp;'
    f'Data: Yahoo Finance (SI=F, GC=F, DXY, US10Y) &amp; Google News RSS'
    f'</div>',
    unsafe_allow_html=True,
)
