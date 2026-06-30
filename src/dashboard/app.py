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


# ── disk helpers ──────────────────────────────────────────────────────────────

def load_latest_briefing() -> tuple[str | None, str | None, dict]:
    today = date.today().isoformat()
    _out = Path(OUTPUTS_DIR)

    # Try flat file first (written by save_outputs as backward-compat)
    flat_files = sorted(_out.glob("briefing_*.txt"))
    if flat_files:
        f = flat_files[-1]
        date_str = f.stem.replace("briefing_", "")
        print(f"[DEBUG] briefing_date={date_str}")
        raw = f.read_text()
        briefing_text, scores = extract_scores(raw)
        print(f"[DEBUG] extract_scores verdict={str(scores.get('verdict', 'MISSING'))[:60]}")
        scores_file = _out / f"scores_{date_str}.json"
        print(f"[DEBUG] flat scores path: {scores_file} exists={scores_file.exists()}")
        if scores_file.exists():
            try:
                scores = json.loads(scores_file.read_text())
                print(f"[DEBUG] flat scores verdict={str(scores.get('verdict', 'MISSING'))[:60]}")
            except Exception as e:
                print(f"[DEBUG] flat scores load error: {e}")
        daily_scores_path = _out / "daily" / date_str / "briefing" / "scores.json"
        print(f"[DEBUG] daily scores path: {daily_scores_path} exists={daily_scores_path.exists()}")
        if daily_scores_path.exists():
            try:
                daily_scores = json.loads(daily_scores_path.read_text())
                print(f"[DEBUG] daily scores verdict={str(daily_scores.get('verdict', 'MISSING'))[:60]}")
            except Exception as e:
                print(f"[DEBUG] daily scores load error: {e}")
        return briefing_text, date_str, scores

    # Fall back to daily structured path
    daily_briefing = _out / "daily" / today / "briefing" / "briefing.txt"
    if daily_briefing.exists():
        raw = daily_briefing.read_text()
        briefing_text, scores = extract_scores(raw)
        scores_file = _out / "daily" / today / "briefing" / "scores.json"
        if scores_file.exists():
            try:
                scores = json.loads(scores_file.read_text())
            except Exception:
                pass
        return briefing_text, today, scores

    return None, None, {}


def load_latest_prices(briefing_date: str | None = None) -> tuple[dict, dict, dict, dict] | None:
    _out = Path(OUTPUTS_DIR)

    # Fall back to daily structured path for the resolved briefing date first —
    # this is the date the rest of the page is actually showing.
    if briefing_date:
        daily_prices = _out / "daily" / briefing_date / "raw" / "prices.json"
        if daily_prices.exists():
            try:
                data = json.loads(daily_prices.read_text())
                return data["silver"], data["gold"], data["dxy"], data["us10y"]
            except Exception:
                pass

    # Try flat file next (legacy compat — may be stale relative to briefing_date)
    flat_files = sorted(_out.glob("prices_*.json"))
    if flat_files:
        try:
            data = json.loads(flat_files[-1].read_text())
            return data["silver"], data["gold"], data["dxy"], data["us10y"]
        except Exception:
            pass

    # Last resort: today's daily structured path
    daily_prices = _out / "daily" / date.today().isoformat() / "raw" / "prices.json"
    if daily_prices.exists():
        try:
            data = json.loads(daily_prices.read_text())
            return data["silver"], data["gold"], data["dxy"], data["us10y"]
        except Exception:
            pass

    return None


def load_latest_history(briefing_date: str | None = None) -> list[dict]:
    _out = Path(OUTPUTS_DIR)

    # Daily structured path for the resolved briefing date first — this is the
    # date the rest of the page is actually showing.
    if briefing_date:
        daily_history = _out / "daily" / briefing_date / "raw" / "history.json"
        if daily_history.exists():
            try:
                return json.loads(daily_history.read_text())
            except Exception:
                pass

    # Try flat file next (legacy compat — may be stale relative to briefing_date)
    flat_files = sorted(_out.glob("history_*.json"))
    if flat_files:
        try:
            return json.loads(flat_files[-1].read_text())
        except Exception:
            pass

    # Last resort: today's daily structured path
    daily_history = _out / "daily" / date.today().isoformat() / "raw" / "history.json"
    if daily_history.exists():
        try:
            return json.loads(daily_history.read_text())
        except Exception:
            pass

    return []


def load_briefing_prices(briefing_date: str) -> dict:
    path = Path(OUTPUTS_DIR) / "daily" / briefing_date / "raw" / "prices.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


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

def escape_dollars(text: str) -> str:
    return text.replace('$', r'\$')


def escape_dollars_safe(text: str) -> str:
    """Escape $ signs in display text but leave markdown link URLs untouched."""
    links: dict[str, str] = {}

    def protect(m: re.Match) -> str:
        key = f"LINK{len(links)}LINK"
        links[key] = m.group(0)
        return key

    protected = re.sub(r'\[([^\]]+)\]\([^)]+\)', protect, text)
    escaped = protected.replace('$', r'\$')
    for key, val in links.items():
        escaped = escaped.replace(key, val)
    return escaped


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

def generate_pdf(
    briefing: str,
    silver: dict,
    gold: dict,
    briefing_date: str,
    scores: dict | None = None,
    dxy: dict | None = None,
    us10y: dict | None = None,
) -> bytes:
    # ── URL stripping (4-pass) ────────────────────────────────────────────────
    clean = re.sub(r'\[([^\]]+)\]\(https?://[^\)]*\)', r'\1', briefing)
    clean = re.sub(r'\[([^\]]+)\]\(https?://[^\s\)]*', r'\1', clean)
    clean = re.sub(r'https?://\S+', '', clean)
    clean = re.sub(r'\(\s*,?\s*\)', '', clean)
    clean = re.sub(r',\s*\)', ')', clean)
    briefing = clean

    if scores is None:
        scores = {}

    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable, SimpleDocTemplate, Paragraph, Spacer,
        Table, TableStyle, PageBreak,
    )

    # ── palette ───────────────────────────────────────────────────────────────
    NAVY      = HexColor("#1a2f5e")
    GREEN     = HexColor("#1a5e3a")
    AMBER     = HexColor("#f0a500")
    GRAY      = HexColor("#888888")
    LGRAY     = HexColor("#f5f7fa")
    BODY      = HexColor("#2c2c2c")
    DGRAY     = HexColor("#d8d8d8")
    NAVY_HEX  = "#1a2f5e"
    GREEN_HEX = "#1a5e3a"
    AMBER_HEX = "#f0a500"
    RED_HEX    = "#a32d2d"
    GRAY_HEX   = "#888888"
    RED        = HexColor("#a32d2d")
    PURPLE_HEX = "#534AB7"
    TEAL_HEX   = "#0f6e56"
    DGRAY2_HEX = "#5F5E5A"
    AMB2_HEX   = "#BA7517"
    LBLUE      = HexColor("#e8ecf4")
    INTERP_BG  = HexColor("#f0f4f8")

    page_w, _ = A4
    margin = 2 * cm
    full_w = page_w - 2 * margin

    # ── style factory (cached to avoid duplicate names per call) ──────────────
    _sc: dict = {}

    def _s(name: str, **kw) -> ParagraphStyle:
        if name not in _sc:
            base = dict(fontName="Helvetica", fontSize=10, leading=14, textColor=BODY)
            base.update(kw)
            _sc[name] = ParagraphStyle(name, **base)
        return _sc[name]

    s_title   = _s("p_title",   fontName="Helvetica-Bold",    fontSize=20, textColor=NAVY,  leading=24, spaceAfter=3)
    s_sub     = _s("p_sub",     fontName="Helvetica-Oblique", fontSize=9,  textColor=GRAY,  leading=12, spaceAfter=2)
    s_body    = _s("p_body",    fontSize=9,  textColor=BODY,  leading=13, spaceAfter=2)
    s_clbl    = _s("p_clbl",    fontName="Helvetica-Bold",    fontSize=7,  textColor=GRAY,  leading=9)
    s_cval    = _s("p_cval",    fontName="Helvetica-Bold",    fontSize=13, textColor=NAVY,  leading=15)
    s_cchg    = _s("p_cchg",    fontName="Helvetica-Bold",    fontSize=8,  leading=10)
    s_dq      = _s("p_dq",      fontSize=8,  textColor=GRAY,  leading=11)
    s_verdict = _s("p_verdict", fontName="Helvetica-Oblique", fontSize=9,  textColor=BODY,  leading=13)
    s_barlbl  = _s("p_barlbl",  fontName="Helvetica-Bold",    fontSize=7,  textColor=GRAY,  leading=9)
    s_barscr  = _s("p_barscr",  fontName="Helvetica-Bold",    fontSize=8,  leading=10)
    s_drvrnk  = _s("p_drvrnk",  fontName="Helvetica-Bold",    fontSize=7,  textColor=GRAY,  leading=9)
    s_drvnm   = _s("p_drvnm",   fontName="Helvetica-Bold",    fontSize=9,  textColor=NAVY,  leading=11)
    s_drvsub    = _s("p_drvsub",  fontSize=7,  textColor=GRAY,  leading=9)

    # Typography hierarchy (PDF redesign)
    s_h1        = _s("p_h1",      fontName="Helvetica-Bold", fontSize=16, textColor=NAVY,
                     spaceBefore=24, spaceAfter=8, leading=20)
    s_h2        = _s("p_h2",      fontName="Helvetica-Bold", fontSize=11, textColor=GREEN,
                     spaceBefore=10, spaceAfter=4, leading=14)
    s_h3        = _s("p_h3",      fontName="Helvetica-Bold", fontSize=10, textColor=NAVY,
                     spaceBefore=8,  spaceAfter=3, leading=13)
    s_interp    = _s("p_interp",  fontName="Helvetica",      fontSize=9.5, textColor=NAVY,
                     leading=13, leftIndent=12, spaceBefore=4, spaceAfter=4)
    s_drv_meta  = _s("p_drvmeta", fontSize=7.5, textColor=HexColor("#666666"), leading=10, spaceAfter=2)
    s_drv_quote = _s("p_drvquote", fontName="Helvetica-Oblique", fontSize=9, textColor=NAVY,
                     leading=12, leftIndent=8, spaceAfter=4)

    # ── utility functions ─────────────────────────────────────────────────────
    def md_to_rl(text: str) -> str:
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"\*(.+?)\*",     r"<i>\1</i>", text)
        text = re.sub(r"[*_`]", "", text)
        return text

    def strip_links(text: str) -> str:
        return re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)

    def score_hex(v) -> str:
        try:
            iv = int(v)
            if iv >= 7: return GREEN_HEX
            if iv >= 4: return AMBER_HEX
            return RED_HEX
        except (ValueError, TypeError):
            return GRAY_HEX

    def score_label(v) -> str:
        try:
            iv = int(v)
            if iv >= 7: return "BULLISH"
            if iv <= 4: return "BEARISH"
            return "NEUTRAL"
        except (ValueError, TypeError):
            return str(v)

    def chg_hex(change: float) -> str:
        return GREEN_HEX if change >= 0 else RED_HEX

    def chg_arrow(change: float) -> str:
        return "▲" if change >= 0 else "▼"

    def draw_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(GRAY)
        canvas.setStrokeColor(DGRAY)
        canvas.setLineWidth(0.3)
        y = margin * 0.55
        canvas.line(margin, y + 11, page_w - margin, y + 11)
        canvas.drawString(margin, y, "Silver Market Intelligence — Confidential")
        canvas.drawRightString(page_w - margin, y, f"Page {doc.page}")
        canvas.restoreState()

    # ── conviction bar (label | filled bar | score) ───────────────────────────
    def conviction_bar(label: str, score, bar_w: float = 130) -> Table:
        try:
            iv = max(1, min(10, int(score)))
        except (ValueError, TypeError):
            iv = 5
        c_hex = score_hex(iv)
        filled = bar_w * iv / 10
        empty  = bar_w - filled
        inner_bar = Table([["", ""]], colWidths=[filled, empty], rowHeights=[8])
        inner_bar.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (0, 0), HexColor(c_hex)),
            ("BACKGROUND",    (1, 0), (1, 0), DGRAY),
            ("TOPPADDING",    (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ]))
        lbl_p = Paragraph(label, s_barlbl)
        scr_p = Paragraph(f'<font color="{c_hex}"><b>{iv}/10</b></font>', s_barscr)
        row = Table([[lbl_p, inner_bar, scr_p]], colWidths=[80, bar_w, 35], rowHeights=[14])
        row.setStyle(TableStyle([
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ("LEFTPADDING",   (0, 0), (-1, -1), 2),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 2),
        ]))
        return row

    # ── metrics strip cell ────────────────────────────────────────────────────
    col5_w = full_w / 5

    def metric_cell(label: str, value: str, change: float, chg_str: str) -> Table:
        c = chg_hex(change)
        a = chg_arrow(change)
        t = Table(
            [
                [Paragraph(label, s_clbl)],
                [Paragraph(value, s_cval)],
                [Paragraph(f'<font color="{c}">{a} {chg_str}</font>', s_cchg)],
            ],
            colWidths=[col5_w - 16],
        )
        t.setStyle(TableStyle([
            ("TOPPADDING",    (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ]))
        return t

    # ── driver card ───────────────────────────────────────────────────────────
    col3_w = full_w / 3
    _RANK_LABELS  = ["#1 PRIMARY", "#2 SECONDARY", "#3 TERTIARY"]
    _RANK_IMPACTS = ["HIGH", "HIGH", "MEDIUM"]

    def driver_card(rank_idx: int, name: str, stars: str, confidence: str) -> Table:
        c_hex = GREEN_HEX if confidence.lower() == "high" else (AMBER_HEX if confidence.lower() == "medium" else RED_HEX)
        star_style = _s(f"p_drvstar{rank_idx}", fontSize=9, textColor=AMBER, leading=11)
        t = Table(
            [
                [Paragraph(_RANK_LABELS[rank_idx], s_drvrnk)],
                [Paragraph(name[:40], s_drvnm)],
                [Paragraph(f"Impact: {_RANK_IMPACTS[rank_idx]}", s_drvsub)],
                [Paragraph(stars or "—", star_style)],
                [Paragraph(f'<font color="{c_hex}">Confidence: {confidence}</font>', s_drvsub)],
            ],
            colWidths=[col3_w - 16],
        )
        t.setStyle(TableStyle([
            ("TOPPADDING",    (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ]))
        return t

    # ── section splitter ──────────────────────────────────────────────────────
    _SECTION_ORDER = [
        "TOP STORIES BY IMPACT",
        "PRICE ACTION SUMMARY",
        "LIMITATIONS TODAY",
        "RANKED MARKET DRIVERS",
        "MARKET DRIVERS",
        "BULL VS BEAR",
        "SUPPLY RISK MONITOR",
        "CONVICTION SCORE",
        "VERDICT",
    ]

    def split_sections(text: str) -> list[tuple[str, str]]:
        positions = []
        for name in _SECTION_ORDER:
            m = re.search(rf"^(?:#{1,3}\s+)?{re.escape(name)}\s*:?\s*$", text, re.MULTILINE)
            if m:
                positions.append((m.start(), m.end(), name))
        positions.sort()
        result = []
        for i, (_, end, name) in enumerate(positions):
            next_start = positions[i + 1][0] if i + 1 < len(positions) else len(text)
            body = text[end:next_start]
            body = re.sub(r"^[\s━─■=\-]+\n", "", body)
            result.append((name, body.strip()))
        return result

    def parse_drivers(briefing: str) -> list[str]:
        matches = re.findall(r'#\d+\s+(.+?)\s*\((?:Primary|Secondary|Tertiary)\)', briefing)
        drivers = []
        for m in matches:
            name = m.strip()
            if name and len(name) > 8:
                drivers.append(name)
        while len(drivers) < 3:
            drivers.append("—")
        return drivers[:3]

    def parse_driver_confidence(briefing: str, one_based_index: int) -> str:
        m = re.search(rf'#\s*{one_based_index}\s+[A-Z]', briefing)
        if not m:
            return "Medium"
        chunk = briefing[m.start(): m.start() + 300]
        conf_m = re.search(r'Confidence:\s*(High|Medium|Low)', chunk, re.IGNORECASE)
        return conf_m.group(1).capitalize() if conf_m else "Medium"

    _ETAG_COLORS = {
        "DATA":      NAVY_HEX,
        "TECHNICAL": TEAL_HEX,
        "MACRO":     PURPLE_HEX,
        "NEWS":      DGRAY2_HEX,
        "INFERENCE": AMB2_HEX,
    }

    def _parse_evidence_tag(text: str) -> tuple[str, str, str]:
        """Returns (color_hex, tag_label, content_without_bracket_tag)."""
        m = re.match(r'^\[(DATA|TECHNICAL|MACRO|NEWS|INFERENCE)\]\s*', text, re.IGNORECASE)
        if m:
            tag = m.group(1).upper()
            return _ETAG_COLORS.get(tag, GRAY_HEX), tag, text[m.end():]
        t = text.lower()
        if any(k in t for k in ("silver:", "gold:", "dxy:", "us10y:", "rsi", "ratio:", "volatility", "$", "bps")):
            return NAVY_HEX, "DATA", text
        if any(k in t for k in ("support", "resistance", "oversold", "overbought", "moving average")):
            return TEAL_HEX, "TECHNICAL", text
        return DGRAY2_HEX, "NEWS", text

    def classify_bullet(text: str) -> tuple[str, str]:
        c, tag, _ = _parse_evidence_tag(text)
        return c, f"[{tag}]"

    # ════════════════════════════════════════════════════════════════════════════
    # PAGE 1 — EXECUTIVE SUMMARY
    # ════════════════════════════════════════════════════════════════════════════
    elems = []

    # Header
    elems.append(Paragraph("Silver Market Intelligence", s_title))
    elems.append(Paragraph(f"Daily Briefing — {briefing_date}", s_sub))
    elems.append(HRFlowable(width="100%", thickness=1.5, color=NAVY, spaceAfter=8, spaceBefore=4))

    # Metrics strip
    ratio_val  = gold["price"] / silver["price"]
    ratio_chg  = ratio_val - 65.0
    _dxy_p     = dxy["price"]              if dxy   else 0.0
    _dxy_chg   = dxy.get("change", 0.0)    if dxy   else 0.0
    _dxy_pct   = dxy.get("change_pct", 0.0) if dxy  else 0.0
    _u10_p     = us10y["price"]            if us10y else 0.0
    _u10_chg   = us10y.get("change", 0.0)  if us10y else 0.0
    _u10_bps   = round(_u10_chg * 100)

    metrics_row = [
        metric_cell("SILVER (SI=F)", f"${silver['price']:.2f}",  silver["change"], f"${silver['change']:+.2f} ({silver['change_pct']:+.2f}%)"),
        metric_cell("GOLD (GC=F)",   f"${gold['price']:,.2f}",   gold["change"],   f"${gold['change']:+.2f} ({gold['change_pct']:+.2f}%)"),
        metric_cell("RATIO",         f"{ratio_val:.1f}",          ratio_chg,        f"{ratio_chg:+.1f} vs avg"),
        metric_cell("DXY",           f"{_dxy_p:.2f}",             _dxy_chg,         f"{_dxy_pct:+.2f}%"),
        metric_cell("US10Y",         f"{_u10_p:.2f}%",            _u10_chg,         f"{_u10_bps:+d}bps"),
    ]
    metrics_tbl = Table([metrics_row], colWidths=[col5_w] * 5, rowHeights=[60])
    metrics_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LGRAY),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LINEAFTER",     (0, 0), (3, 0),   0.5, DGRAY),
    ]))
    elems.append(metrics_tbl)
    elems.append(Spacer(1, 0.3 * cm))

    # Significant move banner
    sig_pct = silver.get("change_pct", 0.0)
    if abs(sig_pct) >= 2.0:
        rsi_m = re.search(r"RSI[- ]?(?:14)?[:\s]+([0-9.]+)", briefing, re.IGNORECASE)
        rsi_val = float(rsi_m.group(1)) if rsi_m else None
        rsi_tag = ""
        if rsi_val is not None:
            rsi_lbl = "OVERSOLD" if rsi_val < 30 else ("OVERBOUGHT" if rsi_val > 70 else "")
            rsi_tag = f" — RSI {rsi_val:.1f}{' ' + rsi_lbl if rsi_lbl else ''}"
        sig_sign  = "+" if sig_pct > 0 else ""
        sig_color = GREEN_HEX if sig_pct > 0 else RED_HEX
        banner_txt = f"*** SIGNIFICANT MOVE — Silver {sig_sign}{sig_pct:.2f}%{rsi_tag}"
        banner = Table(
            [[Paragraph(f'<font color="{sig_color}"><b>{banner_txt}</b></font>', s_body)]],
            colWidths=[full_w],
        )
        banner.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), HexColor("#fffbe6")),
            ("BOX",           (0, 0), (-1, -1), 1.0, AMBER),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ]))
        elems.append(banner)
        elems.append(Spacer(1, 0.25 * cm))

    # Top 3 Market Drivers
    elems.append(Paragraph(
        "<b>Top Market Drivers</b>",
        _s("p_drvtitle", fontName="Helvetica-Bold", fontSize=10, textColor=NAVY, leading=13, spaceAfter=4),
    ))
    drivers = parse_drivers(briefing)  # returns list[str], already padded to 3
    confidences = [parse_driver_confidence(briefing, i + 1) for i in range(3)]

    drv_cells = [driver_card(i, name, "", confidences[i]) for i, name in enumerate(drivers)]
    drv_tbl = Table([drv_cells], colWidths=[col3_w] * 3, rowHeights=[76])
    drv_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LGRAY),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LINEAFTER",     (0, 0), (1, 0),   0.5, DGRAY),
    ]))
    elems.append(drv_tbl)
    elems.append(Spacer(1, 0.3 * cm))

    # Market Snapshot
    elems.append(Paragraph(
        "<b>Market Snapshot</b>",
        _s("p_snaptitle", fontName="Helvetica-Bold", fontSize=10, textColor=NAVY, leading=13, spaceAfter=4),
    ))
    reasons = extract_component_reasons(briefing)

    # RSI for technicals snapshot fallback
    _rsi_brf  = re.search(r"RSI[- ]?(?:14)?[:\s]+([0-9.]+)", briefing, re.IGNORECASE)
    _rsi_v    = float(_rsi_brf.group(1)) if _rsi_brf else None
    _rsi_desc = (
        f"RSI {_rsi_v:.0f} — {'oversold' if _rsi_v < 30 else 'overbought' if _rsi_v > 70 else 'neutral'}"
        if _rsi_v else "Price momentum signals"
    )
    _snap_fallbacks = {
        "macro":       "Yields and USD direction",
        "technicals":  _rsi_desc,
        "supply_risk": "No disruptions detected",
        "sentiment":   "Based on news flow analysis",
    }

    def _snap_status(key: str) -> tuple[str, str]:
        try:
            iv = int(scores.get(key, 5))
        except (ValueError, TypeError):
            iv = 5
        if iv <= 3: return "BEARISH", RED_HEX
        if iv <= 6: return "NEUTRAL", AMBER_HEX
        return "BULLISH", GREEN_HEX

    def _supply_st() -> tuple[str, str]:
        lvl = str(scores.get("supply_risk", "LOW")).upper()
        if lvl == "HIGH":   return "HIGH",   RED_HEX
        if lvl == "MEDIUM": return "MEDIUM", AMBER_HEX
        return "LOW", GREEN_HEX

    snap_rows_cfg = [
        ("MACRO",       _snap_status("macro"),      (reasons.get("macro") or _snap_fallbacks["macro"])[:90]),
        ("TECHNICALS",  _snap_status("technicals"), (reasons.get("technicals") or _snap_fallbacks["technicals"])[:90]),
        ("SUPPLY RISK", _supply_st(),               (reasons.get("supply_risk") or _snap_fallbacks["supply_risk"])[:90]),
        ("SENTIMENT",   _snap_status("sentiment"),  (reasons.get("sentiment") or _snap_fallbacks["sentiment"])[:90]),
    ]
    snap_data = []
    for lbl, (status, st_hex), reason in snap_rows_cfg:
        tag = lbl.replace(" ", "_")
        snap_data.append([
            Paragraph(f"<b>{lbl}</b>",                                          _s(f"p_sl_{tag}", fontName="Helvetica-Bold", fontSize=8, textColor=BODY, leading=10)),
            Paragraph(f'<font color="{st_hex}"><b>{status}</b></font>',         _s(f"p_ss_{tag}", fontName="Helvetica-Bold", fontSize=8, leading=10)),
            Paragraph(reason or "—",                                            _s(f"p_sr_{tag}", fontSize=7, textColor=GRAY, leading=9)),
        ])
    snap_tbl = Table(snap_data, colWidths=[72, 60, full_w - 132], rowHeights=[26] * 4)
    snap_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LGRAY),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW",     (0, 0), (-1, -2), 0.3, DGRAY),
    ]))
    elems.append(snap_tbl)
    elems.append(Spacer(1, 0.25 * cm))

    # Overall Conviction
    overall = int(scores.get("overall", 5)) if scores else 5
    o_hex    = score_hex(overall)
    o_status = score_label(overall)
    elems.append(Paragraph(
        f'<font color="{o_hex}"><b>OVERALL CONVICTION: {overall}/10 — {o_status}</b></font>',
        _s("p_ovtitle", fontName="Helvetica-Bold", fontSize=10, leading=13, spaceAfter=3),
    ))
    elems.append(conviction_bar("Overall", overall, bar_w=200))
    elems.append(Spacer(1, 0.2 * cm))

    # Data Quality line
    dq_m = re.search(r"DATA AVAILABILITY[^\n]*\n(.*?)(?=\nMETHODOLOGY|\n━|\Z)", briefing, re.DOTALL)
    if dq_m:
        dq_raw        = dq_m.group(1)
        avail_items   = re.findall(r"✅\s*([^\n]+)", dq_raw)
        partial_items = re.findall(r"⚠\s*([^\n]+)", dq_raw)
        missing_items = re.findall(r"❌\s*([^\n]+)", dq_raw)
        rel_m2 = re.search(r"reliability[^:]*:\s*(\w+)", dq_raw, re.IGNORECASE)
        rel = rel_m2.group(1) if rel_m2 else "MEDIUM"
        def _dq_esc(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        avail_str   = " ".join(f'<font color="{GREEN_HEX}">[OK]</font> {_dq_esc(x.split("(")[0].strip())}' for x in avail_items[:5])
        partial_str = " ".join(f'<font color="{AMBER_HEX}">[!]</font> {_dq_esc(x.split("(")[0].strip())}' for x in partial_items[:2])
        missing_str = " ".join(f'<font color="{RED_HEX}">[X]</font> {_dq_esc(x.split("(")[0].strip())}' for x in missing_items[:3])
        dq_line = f"Data Quality: {rel} | {avail_str} {partial_str} {missing_str}"
    else:
        dq_line = (
            f'Data Quality: MEDIUM | <font color="{GREEN_HEX}">[OK]</font> Silver '
            f'<font color="{GREEN_HEX}">[OK]</font> Gold '
            f'<font color="{GREEN_HEX}">[OK]</font> Ratio '
            f'<font color="{AMBER_HEX}">[!]</font> ETF Flows '
            f'<font color="{RED_HEX}">[X]</font> COT '
            f'<font color="{RED_HEX}">[X]</font> COMEX'
        )
    elems.append(Paragraph(dq_line, s_dq))
    elems.append(Spacer(1, 0.2 * cm))

    # AI Morning Brief
    verdict      = scores.get("verdict", "") if scores else ""
    morning_brief = verdict if verdict else "Analysis generated from quantitative signals and news flow."
    sentences    = re.split(r'(?<=[.!?])\s+', morning_brief)
    verdict_txt  = " ".join(sentences[:3])
    brief_box = Table(
        [[Paragraph(md_to_rl(verdict_txt),
            _s("p_p1verdict", fontName="Helvetica-Bold", fontSize=10.5, textColor=NAVY, leading=15))]],
        colWidths=[full_w],
    )
    brief_box.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LBLUE),
        ("BOX",           (0, 0), (-1, -1), 1.0, NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
    ]))
    elems.append(brief_box)
    elems.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════════
    # PAGE 2 — DETAILED ANALYSIS (compact, 2-page fit)
    # ════════════════════════════════════════════════════════════════════════════
    elems.append(Paragraph(
        "Detailed Analysis",
        _s("p_p2h", fontName="Helvetica-Bold", fontSize=13, textColor=NAVY, leading=17, spaceAfter=4),
    ))
    elems.append(HRFlowable(width="100%", thickness=1.0, color=NAVY, spaceAfter=6, spaceBefore=2))

    # Compact styles for page 2 (8pt body, 10pt section headers)
    s_p2_sec  = _s("p_p2sec",  fontName="Helvetica-Bold", fontSize=10, textColor=GREEN, leading=13, spaceBefore=4, spaceAfter=1)
    s_p2_body = _s("p_p2body", fontSize=8,  textColor=BODY,  leading=11, spaceAfter=1)
    s_p2_blt  = _s("p_p2blt",  fontSize=8,  textColor=BODY,  leading=11, leftIndent=10, spaceAfter=1)
    s_p2_lbl  = _s("p_p2lbl",  fontName="Helvetica-Bold", fontSize=8,  textColor=NAVY,  leading=11)
    s_p2_drv  = _s("p_p2drv",  fontName="Helvetica-Bold", fontSize=9,  textColor=NAVY,  leading=12)
    s_p2_sub  = _s("p_p2sub",  fontName="Helvetica-Bold", fontSize=8,  textColor=NAVY,  leading=11)
    s_p2_star = _s("p_p2star", fontSize=9,  textColor=AMBER, leading=11)

    def render_section(name: str, body: str, sec_mode: str = "full") -> list:
        """Render one briefing section. sec_mode: full | bull_bear | verdict | limitations | scores_only | one_line | stories | bullets_only"""
        out: list = []
        out.append(Paragraph(name, s_h1))
        out.append(HRFlowable(width="100%", thickness=2, color=GREEN, spaceAfter=16, spaceBefore=2))

        # one_line: INTERPRETATION line, else first non-empty non-separator line
        if sec_mode == "one_line":
            interp_m = re.search(r"INTERPRETATION:\s*(.+?)(?:\n|$)", body, re.IGNORECASE)
            if interp_m:
                out.append(Paragraph(md_to_rl(strip_links(interp_m.group(1).strip())), s_p2_body))
            else:
                for line in body.splitlines():
                    s = line.strip()
                    if s and not re.match(r'^[━─■=\-]{4,}$', s):
                        out.append(Paragraph(md_to_rl(strip_links(s)), s_p2_body))
                        break
            return out

        # stories: numbered items only, max 5, with hyperlinks where available
        if sec_mode == "stories":
            count = 0
            for line in body.splitlines():
                num_m = re.match(r"^(\d+)\.\s+(.+)", line.strip())
                if num_m and count < 5:
                    item_text = num_m.group(2)
                    link_m = re.match(r'\[([^\]]+)\]\(([^)]+)\)(.*)', item_text)
                    if link_m:
                        title = link_m.group(1).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                        url   = link_m.group(2)
                        rest  = md_to_rl(strip_links(link_m.group(3)))
                        out.append(Paragraph(
                            f'<b>{num_m.group(1)}.</b> <link href="{url}">{title}</link>{rest}',
                            s_p2_body,
                        ))
                    else:
                        out.append(Paragraph(f"<b>{num_m.group(1)}.</b> {md_to_rl(strip_links(item_text))}", s_p2_body))
                    count += 1
            return out

        # scores_only: conviction bar lines only
        if sec_mode == "scores_only":
            for line in body.splitlines():
                sc_m = re.match(
                    r"(Macro|Technicals|Sentiment|ETF Flows|Industrial Demand):\s*(\d+)/10",
                    line.strip(), re.IGNORECASE,
                )
                if sc_m:
                    out.append(conviction_bar(sc_m.group(1), int(sc_m.group(2)), bar_w=130))
            return out

        # limitations: plain line list — never apply evidence tag detection
        if sec_mode == "limitations":
            for line in body.splitlines():
                s = re.sub(r'^\[NEWS\]\s+', '', line.strip())
                if not s or re.match(r'^[━─■=\-]{4,}$', s):
                    continue
                s = re.sub(r'^[-*•]\s+', '', s)  # strip any bullet prefix
                out.append(Paragraph(md_to_rl(s), s_body))
            return out

        # verdict: light navy highlighted box
        if sec_mode == "verdict":
            verdict_text = body.strip()
            if verdict_text:
                vbox = Table(
                    [[Paragraph(md_to_rl(strip_links(verdict_text)),
                        _s("p_verdict_box", fontName="Helvetica-Bold", fontSize=10.5,
                           textColor=NAVY, leading=15))]],
                    colWidths=[full_w],
                )
                vbox.setStyle(TableStyle([
                    ("BACKGROUND",    (0, 0), (-1, -1), LBLUE),
                    ("BOX",           (0, 0), (-1, -1), 1.0, NAVY),
                    ("TOPPADDING",    (0, 0), (-1, -1), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
                ]))
                out.append(Spacer(1, 0.3 * cm))
                out.append(vbox)
            return out

        # bull_bear: 2-column green/red layout
        if sec_mode == "bull_bear":
            bear_pat = re.compile(r'###?\s*BEARISH CASE', re.IGNORECASE)
            halves   = bear_pat.split(body, maxsplit=1)
            bull_body_bb = halves[0]
            bear_body_bb = halves[1] if len(halves) > 1 else ""

            def _side_paras(text: str, side: str) -> list:
                paras = []
                blt_st = _s(f"p_bb_{side}_b", fontSize=8, textColor=BODY, leading=11, leftIndent=8, spaceAfter=1)
                itp_st = _s(f"p_bb_{side}_i", fontName="Helvetica-Oblique", fontSize=8.5,
                            textColor=NAVY, leading=12, leftIndent=8, spaceAfter=2)
                for ln in text.splitlines():
                    s = re.sub(r'^#{1,3}\s+', '', ln.strip())
                    if not s or re.match(r'^[━─■=\-]{4,}$', s):
                        continue
                    if re.match(r"^BULLISH CASE", s, re.IGNORECASE):
                        continue
                    if re.match(r"^EVIDENCE:", s, re.IGNORECASE):
                        paras.append(Paragraph("<b>EVIDENCE:</b>", s_h2))
                    elif re.match(r"^[-*•]\s", s):
                        content = strip_links(s[2:].strip())
                        c, tag, rest = _parse_evidence_tag(content)
                        paras.append(Paragraph(f'<font color="{c}"><b>[{tag}]</b></font> {md_to_rl(rest)}', blt_st))
                    elif re.match(r"^INTERPRETATION:", s, re.IGNORECASE):
                        paras.append(Paragraph(md_to_rl(strip_links(s[15:].strip())), itp_st))
                return paras or [Paragraph("—", s_body)]

            col_w_bb = (full_w - 6) / 2

            def _col_table_bb(hdr_para, paras, accent_hex) -> Table:
                rows = [[hdr_para]] + [[p] for p in paras]
                t = Table(rows, colWidths=[col_w_bb])
                t.setStyle(TableStyle([
                    ("TOPPADDING",    (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
                    ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                    ("BACKGROUND",    (0, 0), (0, 0), HexColor("#f5f5f5")),
                    ("LINEBELOW",     (0, 0), (0, 0), 1.5, HexColor(accent_hex)),
                ]))
                return t

            bull_hdr = Paragraph(
                f'<font color="{GREEN_HEX}"><b>▲ BULLISH CASE</b></font>',
                _s("p_bb_bull_hdr", fontName="Helvetica-Bold", fontSize=10, textColor=HexColor(GREEN_HEX), leading=13),
            )
            bear_hdr = Paragraph(
                f'<font color="{RED_HEX}"><b>▼ BEARISH CASE</b></font>',
                _s("p_bb_bear_hdr", fontName="Helvetica-Bold", fontSize=10, textColor=HexColor(RED_HEX), leading=13),
            )
            bb_outer = Table(
                [[_col_table_bb(bull_hdr, _side_paras(bull_body_bb, "bull"), GREEN_HEX),
                  _col_table_bb(bear_hdr, _side_paras(bear_body_bb, "bear"), RED_HEX)]],
                colWidths=[col_w_bb, col_w_bb],
            )
            bb_outer.setStyle(TableStyle([
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ("LINEAFTER",     (0, 0), (0, -1), 0.5, HexColor("#d0d0d0")),
                ("TOPPADDING",    (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ("LEFTPADDING",   (0, 0), (-1, -1), 0),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
            ]))
            out.append(bb_outer)
            return out

        is_conv = (name == "CONVICTION SCORE")

        for line in body.splitlines():
            stripped = re.sub(r'^#{1,3}\s+', '', line.strip())
            if not stripped:
                out.append(Spacer(1, 1))
                continue
            if re.match(r'^[━─■=\-]{4,}$', stripped):
                continue

            # bullets_only mode
            if sec_mode == "bullets_only":
                if re.match(r"^[-*•]\s", stripped):
                    content = strip_links(stripped[2:].strip())
                    c, tag, rest = _parse_evidence_tag(content)
                    out.append(Paragraph(
                        f'<font color="{c}"><b>[{tag}]</b></font> {md_to_rl(rest)}',
                        s_p2_blt,
                    ))
                elif re.match(r"^(BULLISH CASE|BEARISH CASE|VERDICT)", stripped, re.IGNORECASE):
                    out.append(Spacer(1, 2))
                    c_h = GREEN_HEX if "BULLISH" in stripped.upper() else (RED_HEX if "BEARISH" in stripped.upper() else NAVY_HEX)
                    label = ' '.join(stripped.split()[:2]) if 'CASE' in stripped else stripped.split()[0]
                    out.append(Paragraph(f'<font color="{c_h}"><b>{label}</b></font>', s_h3))
                elif (cm2 := re.match(r"CONFIDENCE:\s*(High|Medium|Low)\s*[—–\-]+\s*(.+)", stripped, re.IGNORECASE)):
                    lvl = cm2.group(1)
                    c_h = GREEN_HEX if lvl.lower() == "high" else (AMBER_HEX if lvl.lower() == "medium" else RED_HEX)
                    out.append(Paragraph(
                        f'<b>CONFIDENCE:</b> <font color="{c_h}"><b>[{lvl.upper()}]</b></font>  {md_to_rl(strip_links(cm2.group(2).strip()))}',
                        s_body,
                    ))
                continue

            # EVIDENCE:
            if re.match(r"^EVIDENCE:", stripped, re.IGNORECASE):
                rest = stripped[9:].strip()
                out.append(Paragraph(
                    f"<b>EVIDENCE:</b>{' ' + md_to_rl(strip_links(rest)) if rest else ''}",
                    s_h2,
                ))
                continue

            # INTERPRETATION: — highlighted box
            if re.match(r"^INTERPRETATION:", stripped, re.IGNORECASE):
                rest = stripped[15:].strip()
                ibox = Table(
                    [[Paragraph(md_to_rl(strip_links(rest)), s_interp)]],
                    colWidths=[full_w],
                )
                ibox.setStyle(TableStyle([
                    ("BACKGROUND",    (0, 0), (-1, -1), INTERP_BG),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
                    ("TOPPADDING",    (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                out.append(ibox)
                continue

            # Net impact:
            if re.match(r"^-?\s*Net impact:", stripped, re.IGNORECASE):
                out.append(Paragraph(f"<i>{md_to_rl(strip_links(stripped))}</i>", s_body))
                continue

            # CONFIDENCE: — colored badge
            conf_m = re.match(r"CONFIDENCE:\s*(High|Medium|Low)\s*[—–\-]+\s*(.+)", stripped, re.IGNORECASE)
            if conf_m:
                lvl     = conf_m.group(1)
                rsn_txt = conf_m.group(2).strip()
                c_hex   = GREEN_HEX if lvl.lower() == "high" else (AMBER_HEX if lvl.lower() == "medium" else RED_HEX)
                out.append(Paragraph(
                    f'<b>CONFIDENCE:</b> <font color="{c_hex}"><b>[{lvl.upper()}]</b></font>  {md_to_rl(strip_links(rsn_txt))}',
                    s_body,
                ))
                continue

            # Star ratings
            if re.search(r'[★☆]{3,}', stripped):
                out.append(Paragraph(stripped, _s("p_stars", fontSize=9, textColor=AMBER, leading=11)))
                continue

            # Impact/Confidence/Evidence header (driver cards)
            if re.match(r"^Impact:\s*(HIGH|MEDIUM|LOW)", stripped, re.IGNORECASE):
                out.append(Paragraph(stripped, s_drv_meta))
                continue

            # Quoted sentence (driver cards) — italic navy
            if stripped.startswith('"') and stripped.endswith('"'):
                out.append(Paragraph(f"<i>{md_to_rl(strip_links(stripped[1:-1]))}</i>", s_drv_quote))
                continue

            # Conviction bars
            if is_conv:
                sc_m = re.match(
                    r"\*?\*?(Macro|Technicals|Sentiment|ETF Flows|Industrial Demand):\s*(\d+)/10",
                    stripped, re.IGNORECASE,
                )
                if sc_m:
                    out.append(conviction_bar(sc_m.group(1), int(sc_m.group(2)), bar_w=130))
                    continue

            # Bullets — evidence tag coloring
            if re.match(r"^[-*•]\s", stripped):
                content = strip_links(stripped[2:].strip())
                c, tag, rest = _parse_evidence_tag(content)
                out.append(Paragraph(
                    f'<font color="{c}"><b>[{tag}]</b></font> {md_to_rl(rest)}',
                    s_p2_blt,
                ))
                continue

            # Driver headers — Level 3
            if re.match(r"^#\d+\s+[A-Z]", stripped):
                out.append(Spacer(1, 4))
                out.append(Paragraph(f"<b>{md_to_rl(stripped)}</b>", s_h3))
                continue

            # ALL-CAPS sub-headers — Level 2
            if re.match(r"^[A-Z][A-Z\s\/]{4,}$", stripped):
                out.append(Spacer(1, 2))
                out.append(Paragraph(f"<b>{stripped}</b>", s_h2))
                continue

            # Numbered items
            num_m = re.match(r"^(\d+)\.\s+(.+)", stripped)
            if num_m:
                out.append(Paragraph(f"<b>{num_m.group(1)}.</b> {md_to_rl(strip_links(num_m.group(2)))}", s_body))
                continue

            # Key: value
            kv_m = re.match(r"^([A-Za-z][A-Za-z\s]{2,28}):\s+(.+)", stripped)
            if kv_m:
                out.append(Paragraph(
                    f"<b>{md_to_rl(kv_m.group(1))}:</b> {md_to_rl(strip_links(kv_m.group(2)))}",
                    s_body,
                ))
                continue

            # Default
            out.append(Paragraph(md_to_rl(strip_links(stripped)), s_body))

        return out

    # Sections to render on page 2 and their render mode
    _P2_RENDER = {
        "TOP STORIES BY IMPACT":  "stories",
        "PRICE ACTION SUMMARY":   "full",
        "LIMITATIONS TODAY":      "limitations",
        "RANKED MARKET DRIVERS":  "full",
        "MARKET DRIVERS":         "full",
        "BULL VS BEAR":           "bull_bear",
        "SUPPLY RISK MONITOR":    "one_line",
        "CONVICTION SCORE":       "scores_only",
        "VERDICT":                "verdict",
    }

    sections = split_sections(briefing)
    if len(sections) < 3:
        # Fallback: strip noise blocks and render full text
        p2_text = re.sub(
            r'DATA AVAILABILITY.*?(?=TOP STORIES|PRICE ACTION|\Z)',
            '', briefing, flags=re.DOTALL | re.IGNORECASE,
        )
        p2_text = re.sub(
            r'METHODOLOGY:.*?(?=TOP STORIES|PRICE ACTION|\Z)',
            '', p2_text, flags=re.DOTALL | re.IGNORECASE,
        )
        elems.extend(render_section("FULL BRIEFING", p2_text))
    else:
        for sec_name, sec_body in sections:
            mode = _P2_RENDER.get(sec_name, "full")
            elems.extend(render_section(sec_name, sec_body, sec_mode=mode))

    # ── Build ─────────────────────────────────────────────────────────────────
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=margin, bottomMargin=margin * 1.6,
        leftMargin=margin, rightMargin=margin,
    )
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


def render_silver_chart(history: list[dict]) -> go.Figure:
    dates = [h.get("date", "") for h in history]
    closes = [h.get("close", 0) for h in history]
    color = "#00d4aa" if closes[-1] >= closes[0] else "#ff4757"
    high_30d = max(closes)
    low_30d = min(closes)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=closes,
        mode="lines",
        line=dict(color=color, width=2),
        hovertemplate="$%{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=high_30d, line_dash="dot", line_color="#1a2035", line_width=1)
    fig.add_hline(y=low_30d, line_dash="dot", line_color="#1a2035", line_width=1)
    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            showticklabels=True,
            tickformat="%b %d",
            tickfont=dict(color="#5a6a7e", size=10),
            tickcolor="#5a6a7e",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#1a2035",
            showticklabels=True,
            tickformat="$,.2f",
            tickfont=dict(color="#5a6a7e", size=10),
            tickcolor="#5a6a7e",
            side="right",
        ),
        height=180,
        margin=dict(l=0, r=60, t=20, b=30),
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
    )
    return fig


def parse_story_list(briefing: str) -> list[dict]:
    section = extract_top_stories(briefing) or briefing
    stories = []
    for m in re.finditer(r"\d+\.\s+\[([^\]]+)\]\(([^)]*(?:\([^)]*\))*[^)]*)\)\s*[—–\-]+\s*(.+)", section):
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
    for key, hdr_pat in [
        ("macro",       r"Macro:\s*\d+/10"),
        ("technicals",  r"Technicals:\s*\d+/10"),
        ("sentiment",   r"Sentiment:\s*\d+/10"),
        ("supply_risk", r"Risk:\s*(?:LOW|MEDIUM|HIGH)"),
    ]:
        hdr_m = re.search(hdr_pat, text, re.IGNORECASE)
        if not hdr_m:
            result[key] = ""
            continue
        chunk = text[hdr_m.start(): hdr_m.start() + 600]
        if key == "supply_risk":
            sub_m = re.search(r"CONFIDENCE:\s*\w+\s*[—–\-]+\s*(.+?)(?:\n|$)", chunk, re.IGNORECASE)
            if not sub_m:
                sub_m = re.search(r"INTERPRETATION:\s*(.+?)(?:\n|$)", chunk, re.IGNORECASE)
        else:
            sub_m = re.search(r"INTERPRETATION:\s*(.+?)(?:\n|$)", chunk, re.IGNORECASE)
        result[key] = sub_m.group(1).strip() if sub_m else ""
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
.block-container { padding: 0.25rem 1.5rem 1rem 1.5rem !important; }

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

/* Full briefing expander styling */
.briefing-content h2 {
    font-size: 1.3rem;
    font-weight: 700;
    color: #1a2f5e;
    margin-top: 1.5rem;
    border-bottom: 2px solid #1a5e3a;
    padding-bottom: 4px;
}
.briefing-content h3 {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1a2f5e;
    margin-top: 1rem;
}
.briefing-content strong {
    color: #1a2f5e;
}

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
div[data-testid="stDownloadButton"] button {
    background-color: #1a2f5e !important;
    color: white !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    border: none !important;
    width: 100% !important;
}
div[data-testid="stDownloadButton"] button:hover {
    background-color: #1a5e3a !important;
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

/* ── compact layout ──────────────────────────────────────────────────── */
html, body, [class*="css"] { font-size: 13px !important; }

[data-testid="metric-container"] { padding: 0.3rem 0.5rem !important; }
[data-testid="stMetricValue"] { font-size: 1.4rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

[data-testid="block-container"] { padding-top: 1rem !important; padding-bottom: 0.5rem !important; }
.element-container { margin-bottom: 0.25rem !important; }

[data-testid="stSidebar"] { font-size: 0.8rem !important; }

h1 { font-size: 1.3rem !important; margin-bottom: 0.3rem !important; }
h2 { font-size: 1.1rem !important; margin-bottom: 0.2rem !important; }
h3 { font-size: 1rem !important; margin-bottom: 0.2rem !important; }

/* Smaller chart area */
[data-testid="stPlotlyChart"] { margin-bottom: 0 !important; }

/* Compact morning brief */
.morning-brief-box { padding: 0.5rem 1rem !important; font-size: 0.85rem !important; }

/* Pull full analysis section up */
.full-analysis-section { margin-top: 0.25rem !important; }

/* Sidebar title larger */
[data-testid="stSidebar"] .sidebar-title { font-size: 1.2rem !important; font-weight: bold !important; }

/* Fix clipped sparklines in metric cards */
[data-testid="metric-container"] { overflow: visible !important; height: auto !important; }

/* Smaller alert banner */
.alert-banner { font-size: 0.75rem !important; padding: 0.3rem 0.75rem !important; }

/* Compact section headers */
.section-header { font-size: 0.7rem !important; letter-spacing: 0.08em !important; padding: 0.2rem 0 !important; }
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

with st.sidebar:
    st.markdown(
        '<div style="padding:20px 16px 14px;border-bottom:1px solid #1a2035;">'
        '<div style="font-size:1.2rem;font-weight:800;color:#ffffff;letter-spacing:-0.01em;">'
        '🪙 Silver Market Intelligence</div>'
        '<div style="font-size:0.7rem;color:#4a5a72;margin-top:3px;">AI-powered metals briefing</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="padding:12px 16px;margin-top:4px;">', unsafe_allow_html=True)
    last_updated_slot = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)


# ── load / generate briefing ──────────────────────────────────────────────────

briefing, briefing_date, scores = load_latest_briefing()
_prices = load_latest_prices(briefing_date)
if _prices is not None:
    silver, gold, dxy, us10y = _prices
else:
    silver = gold = dxy = us10y = None
history = load_latest_history(briefing_date)

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
            f'padding:14px 14px 10px;position:relative;min-height:86px;overflow:visible;">'
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
            f'padding:3px 18px;display:flex;align-items:center;justify-content:space-between;">'
            f'<div>'
            f'<span style="color:#ffa500;margin-right:8px;font-size:0.85rem;">⚡</span>'
            f'<span style="font-size:0.75rem;font-weight:700;color:{_fg};text-transform:uppercase;'
            f'letter-spacing:0.06em;">SIGNIFICANT MOVE DETECTED</span>'
            f'<span style="font-size:0.75rem;color:#c8d4e8;margin-left:10px;">'
            f'Silver {_sign}{_sig_pct:.2f}% — briefing focused on move drivers</span>'
            f'</div>'
            f'<span style="font-size:0.72rem;color:{_fg};white-space:nowrap;cursor:pointer;">'
            f'View details →</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)

# ── Row 4: 30-day chart + Market Snapshot ────────────────────────────────────

chart_col, snap_col = st.columns([13, 7])

with chart_col:
    st.markdown(
        '<div style="font-size:0.65rem;font-weight:700;color:#4a5a72;text-transform:uppercase;'
        'letter-spacing:0.09em;margin-bottom:6px;">Silver Price — 30 Days</div>',
        unsafe_allow_html=True,
    )
    if history:
        st.plotly_chart(render_silver_chart(history), use_container_width=True)
    else:
        st.markdown(
            '<div style="background:#0d1117;border:1px solid #1a2035;border-radius:6px;'
            'padding:30px 18px;color:#4a5a72;font-size:0.8rem;text-align:center;">'
            'Refresh to load chart data</div>',
            unsafe_allow_html=True,
        )

with snap_col:
    # ── Market Snapshot ───────────────────────────────────────────────────────
    reasons = extract_component_reasons(briefing)

    _rsi_m = re.search(r"RSI[- ]?(?:14)?[:\s]+([0-9.]+)", briefing, re.IGNORECASE)
    _rsi = float(_rsi_m.group(1)) if _rsi_m else 50.0
    _rsi_label = "oversold" if _rsi < 30 else ("overbought" if _rsi > 70 else "neutral")

    _macro_score = int(scores.get("macro", 5))
    _tech_score  = int(scores.get("technicals", 5))
    _sent_score  = int(scores.get("sentiment", 5))

    _snap_rows = [
        ("Macro",       score_to_status(_macro_score),
         reasons.get("macro")      or f"Score {_macro_score}/10 — macro conditions"),
        ("Technicals",  score_to_status(_tech_score),
         reasons.get("technicals") or f"RSI {_rsi:.0f} — {_rsi_label}"),
        ("Supply Risk", supply_status(scores.get("supply_risk", "LOW")),
         reasons.get("supply_risk") or extract_supply_risk_reason(briefing) or "No disruptions detected"),
        ("Sentiment",   score_to_status(_sent_score),
         reasons.get("sentiment")  or f"Score {_sent_score}/10 — mixed signals"),
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
            f'<span style="font-size:0.6rem;color:#4a5a72;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:0.07em;">{label}</span>'
            f'<span style="font-size:0.6rem;color:{status_color};font-weight:700;'
            f'letter-spacing:0.06em;">{status}</span>'
            f'</div>'
            f'<div style="font-size:0.7rem;color:#4a5a72;line-height:1.4;">{short_reason}</div>'
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
        f'font-size:1rem;font-weight:800;color:{o_color};font-family:\'SF Mono\',monospace;">'
        f'{overall}/10</div>'
        f'<span style="background:{o_color}22;color:{o_color};font-size:0.67rem;font-weight:700;'
        f'padding:2px 10px;border-radius:3px;letter-spacing:0.07em;">{o_status}</span>'
        f'</div>'
        '</div>',
        unsafe_allow_html=True,
    )

# ── Row 5: AI Morning Brief ───────────────────────────────────────────────────

verdict = scores.get("verdict", "Analysis generated from market signals and news flow.")

st.markdown(f"""
<div style="border:1px solid #2a9d8f;border-radius:8px;padding:0.6rem 1rem;background:#0d1f1e;">
    <div style="font-size:0.65rem;color:#2a9d8f;letter-spacing:0.1em;margin-bottom:0.3rem;">
        AI MORNING BRIEF — {briefing_date}
    </div>
    <div style="font-size:0.9rem;color:#e9c46a;font-style:italic;line-height:1.4;">
        {verdict}
    </div>
</div>
""", unsafe_allow_html=True)

# ── Row 6: Full Analysis + Methodology (fixed-height columns) ────────────────

analysis_col, method_col = st.columns([3, 2])

with analysis_col:
    st.markdown(
        '<div style="font-size:0.6rem;font-weight:700;color:#4a5a72;text-transform:uppercase;'
        'letter-spacing:0.08em;margin-bottom:3px;padding:0.1rem 0;">Full Analysis</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="height:420px;overflow-y:auto;background:#0d1117;border:1px solid #1a2035;'
        'border-radius:8px;padding:16px 20px;">'
        + escape_dollars_safe(briefing)
        + '</div>',
        unsafe_allow_html=True,
    )

with method_col:
    st.markdown(
        '<div style="font-size:0.6rem;font-weight:700;color:#4a5a72;text-transform:uppercase;'
        'letter-spacing:0.08em;margin-bottom:3px;padding:0.1rem 0;">About This Report</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="height:auto;background:#0d1117;border:1px solid #1a5e3a;border-radius:8px;'
        'padding:8px 12px;margin-bottom:12px;">'
        '<p style="color:#00d4aa;font-weight:600;margin-bottom:8px;font-size:0.9rem;">'
        'How this analysis is generated</p>'
        '<div style="font-size:0.8rem;color:#8ab58a;line-height:1.8;">'
        '① Live prices fetched from Yahoo Finance (Silver, Gold, DXY, US10Y)<br>'
        '② Signals computed: RSI-14, 30d volatility, significant move detection<br>'
        '③ 20 news articles ingested from Google News, Reuters, Kitco<br>'
        '④ Claude reasons from data first, news second<br>'
        '⑤ Conviction scores extracted as structured JSON'
        '</div></div>',
        unsafe_allow_html=True,
    )
    if silver is not None:
        bp = load_briefing_prices(briefing_date or "")
        pdf_bytes = generate_pdf(
            briefing=briefing,
            silver=bp.get("silver", silver),
            gold=bp.get("gold", gold),
            dxy=bp.get("dxy", dxy),
            us10y=bp.get("us10y", us10y),
            briefing_date=briefing_date or "",
            scores=scores,
        )
        st.markdown("""
<style>
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #d8d8d8, #ffffff) !important;
    color: #0a0a0a !important;
    font-weight: 900 !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.7rem 1.5rem !important;
    box-shadow: 0 4px 25px rgba(255, 255, 255, 0.45) !important;
    width: 100% !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: linear-gradient(135deg, #efefef, #ffffff) !important;
    box-shadow: 0 6px 30px rgba(255, 255, 255, 0.6) !important;
    transform: translateY(-2px) !important;
}
</style>
""", unsafe_allow_html=True)
        st.download_button("⬇ Download Full Briefing PDF", data=pdf_bytes, file_name=f"silver_briefing_{briefing_date}.pdf", mime="application/pdf", use_container_width=True)

# ── footer ────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    f'<div style="font-size:0.72rem;color:#2d3f5a;">'
    f'Briefing date: {briefing_date or "—"}&nbsp;&nbsp;|&nbsp;&nbsp;'
    f'Data: Yahoo Finance (SI=F, GC=F, DXY, US10Y) &amp; Google News RSS'
    f'</div>',
    unsafe_allow_html=True,
)
