"""Shared helpers: theme/CSS, image loading, model loading, prediction."""
import base64
import json
import textwrap
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
IMG_DIR = ROOT / "assets" / "images"
MODEL_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "airline_satisfaction_dataset.csv"

# Exact column order the model was trained on (see train_model.py)
FEATURES = [
    "Gender", "Customer Type", "Age", "Type of Travel", "Class", "Flight Distance",
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Inflight service", "Cleanliness",
    "Departure Delay in Minutes", "Arrival Delay in Minutes",
]
SERVICE_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Inflight service", "Cleanliness",
]

# Palette: apron navy, signage yellow, concourse grey, runway teal, gate red
NAVY, YELLOW, GREY, TEAL, RED = "#0E2238", "#F5B700", "#E9EDF1", "#1F7A8C", "#D64545"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Barlow', 'Segoe UI', sans-serif; color: {NAVY}; }}
h1, h2, h3, h4 {{ font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif !important; color: {NAVY}; letter-spacing: .2px; }}
.stApp {{ background: {GREY}; }}
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }}

/* sidebar = departures concourse */
section[data-testid="stSidebar"] {{ background: {NAVY}; }}
section[data-testid="stSidebar"] * {{ color: #DCE5EE; }}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"],
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {{ border-radius: 6px; font-family: 'Barlow Condensed', sans-serif; font-size: 1.12rem; letter-spacing: .4px; }}
section[data-testid="stSidebar"] a[aria-current="page"] {{ background: {YELLOW} !important; }}
section[data-testid="stSidebar"] a[aria-current="page"] * {{ color: {NAVY} !important; font-weight: 700; }}
.brand {{ display:flex; align-items:center; gap:.6rem; padding:.2rem 0 1rem 0; }}
.brand-mark {{ width:34px; height:34px; border-radius:50%; background:{YELLOW}; display:flex; align-items:center; justify-content:center; color:{NAVY}; font-weight:700; font-size:1.1rem; }}
.brand-name {{ font-family:'Barlow Condensed',sans-serif; font-size:1.5rem; font-weight:700; color:#fff; line-height:1; }}
.brand-tag {{ font-size:.78rem; color:#9FB2C6; }}

/* gate sign page header */
.gate {{ display:flex; align-items:stretch; margin-bottom:1.4rem; border-radius:10px; overflow:hidden; background:{NAVY}; box-shadow:0 6px 18px rgba(14,34,56,.18); }}
.gate-badge {{ background:{YELLOW}; color:{NAVY}; font-family:'Barlow Condensed',sans-serif; font-weight:700; font-size:2.1rem; padding:.7rem 1.3rem; display:flex; flex-direction:column; justify-content:center; line-height:1; min-width:96px; }}
.gate-badge small {{ font-size:.8rem; font-weight:600; letter-spacing:.5px; }}
.gate-text {{ padding:.8rem 1.4rem; }}
.gate-title {{ font-family:'Barlow Condensed',sans-serif; font-size:2.1rem; font-weight:700; color:#fff; line-height:1.05; }}
.gate-sub {{ color:#B5C5D6; font-size:1rem; margin-top:.25rem; max-width:62ch; }}

/* hero */
.hero {{ position:relative; min-height:430px; border-radius:14px; background-size:cover; background-position:center; display:flex; align-items:center; box-shadow:0 14px 34px rgba(14,34,56,.28); overflow:hidden; }}
.hero-copy {{ padding:2.4rem 3rem; max-width:620px; }}
.hero-sign {{ display:inline-block; background:{YELLOW}; color:{NAVY}; font-family:'Barlow Condensed',sans-serif; font-weight:700; font-size:1.05rem; padding:.18rem .7rem; border-radius:4px; margin-bottom:1rem; }}
.hero h1 {{ color:#fff !important; font-size:3.7rem; line-height:.98; margin:0 0 .9rem 0; font-weight:700; }}
.hero p {{ color:#E3EBF3; font-size:1.12rem; line-height:1.5; max-width:48ch; margin:0; }}

/* departures board */
.board {{ background:{NAVY}; border-radius:12px; padding:.5rem 1.2rem 1rem 1.2rem; box-shadow:0 8px 22px rgba(14,34,56,.22); }}
.board-head {{ display:grid; grid-template-columns: 1.4fr 1fr 1.2fr; color:#7F95AB; font-size:.92rem; padding:.7rem 0 .4rem 0; border-bottom:1px solid #243A52; }}
.board-row {{ display:grid; grid-template-columns: 1.4fr 1fr 1.2fr; padding:.62rem 0; border-bottom:1px solid #1B3047; font-family:'Barlow Condensed',sans-serif; font-size:1.5rem; font-weight:600; color:{YELLOW}; letter-spacing:.8px; text-transform:uppercase; }}
.board-row:last-child {{ border-bottom:none; }}
.board-row .st {{ color:#8EE0A6; }}
.board-row .st.warn {{ color:#FF8A80; }}
.board-row span:nth-child(2) {{ color:#fff; }}

/* cards */
.card {{ background:#fff; border-radius:12px; padding:1.2rem 1.35rem; box-shadow:0 4px 14px rgba(14,34,56,.08); height:100%; }}
.card h4 {{ margin:.1rem 0 .4rem 0; font-size:1.35rem; }}
.card p {{ margin:0; color:#41576D; font-size:.98rem; line-height:1.5; }}
.photo-cap {{ font-size:.85rem; color:#5D7289; margin-top:.35rem; }}
.section-title {{ font-family:'Barlow Condensed',sans-serif; font-size:1.9rem; font-weight:700; margin:2rem 0 .8rem 0; }}

/* boarding pass */
.pass {{ display:flex; background:#fff; border-radius:16px; overflow:hidden; box-shadow:0 14px 34px rgba(14,34,56,.2); margin:.6rem 0 1rem 0; }}
.pass-main {{ flex:1; padding:1.2rem 1.6rem; }}
.pass-top {{ display:flex; justify-content:space-between; align-items:center; border-bottom:2px solid {GREY}; padding-bottom:.6rem; margin-bottom:.9rem; }}
.pass-airline {{ font-family:'Barlow Condensed',sans-serif; font-weight:700; font-size:1.5rem; color:{NAVY}; }}
.pass-type {{ font-family:'Barlow Condensed',sans-serif; font-weight:700; font-size:1.05rem; background:{NAVY}; color:{YELLOW}; padding:.1rem .6rem; border-radius:4px; }}
.pass-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:.9rem 1rem; }}
.pass-grid .l {{ font-size:.78rem; color:#6B8097; }}
.pass-grid .v {{ font-family:'Barlow Condensed',sans-serif; font-size:1.45rem; font-weight:600; color:{NAVY}; line-height:1.1; }}
.pass-verdict {{ margin-top:1.1rem; display:flex; align-items:center; gap:.9rem; }}
.stamp {{ font-family:'Barlow Condensed',sans-serif; font-weight:700; font-size:2.1rem; padding:.05rem .9rem; border:3px solid; border-radius:8px; transform:rotate(-3deg); letter-spacing:1px; }}
.stamp.ok {{ color:{TEAL}; border-color:{TEAL}; }}
.stamp.bad {{ color:{RED}; border-color:{RED}; }}
.pass-note {{ color:#41576D; font-size:.98rem; max-width:34ch; }}
.pass-stub {{ width:230px; background:{NAVY}; color:#fff; padding:1.2rem 1.3rem; display:flex; flex-direction:column; justify-content:space-between; border-left:3px dashed {GREY}; position:relative; }}
.pass-stub .l {{ font-size:.8rem; color:#9FB2C6; }}
.pass-stub .big {{ font-family:'Barlow Condensed',sans-serif; font-size:3.6rem; font-weight:700; color:{YELLOW}; line-height:1; }}
.bar-track {{ height:8px; background:#243A52; border-radius:4px; overflow:hidden; margin:.5rem 0; }}
.bar-fill {{ height:100%; background:{YELLOW}; }}
.barcode {{ height:44px; border-radius:3px; background:repeating-linear-gradient(90deg,#fff 0 2px,transparent 2px 4px,#fff 4px 5px,transparent 5px 9px,#fff 9px 12px,transparent 12px 14px); opacity:.9; }}
@media (max-width: 760px) {{
  .hero h1 {{ font-size:2.5rem; }} .hero-copy {{ padding:1.6rem; }}
  .pass {{ flex-direction:column; }} .pass-stub {{ width:auto; border-left:none; border-top:3px dashed {GREY}; }}
  .pass-grid {{ grid-template-columns:repeat(2,1fr); }}
}}
</style>
"""


def setup(title: str):
    st.set_page_config(page_title=f"SkyPulse | {title}", page_icon="✈️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="brand-mark">✈</div><div>'
            '<div class="brand-name">SkyPulse</div>'
            '<div class="brand-tag">Passenger satisfaction terminal</div></div></div>',
            unsafe_allow_html=True,
        )


def html(block: str):
    """Render an HTML block (dedented, blank lines removed so Markdown can't mangle it)."""
    cleaned = "\n".join(l.strip() for l in textwrap.dedent(block).splitlines() if l.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def gate_header(gate: str, title: str, subtitle: str):
    html(f"""
    <div class="gate"><div class="gate-badge"><small>Gate</small>{gate}</div>
    <div class="gate-text"><div class="gate-title">{title}</div>
    <div class="gate-sub">{subtitle}</div></div></div>
    """)


@st.cache_data(show_spinner=False)
def img_b64(name: str) -> str:
    p = IMG_DIR / name
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""


def photo(name: str, caption: str = ""):
    p = IMG_DIR / name
    if p.exists():
        st.image(str(p), use_container_width=True)
        if caption:
            st.markdown(f'<div class="photo-cap">{caption}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------- data / model
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.drop(columns=[c for c in ["Unnamed: 0", "id", ""] if c in df.columns])
    if "Arrival Delay in Minutes" not in df.columns:
        st.error(
            "data/airline_satisfaction_dataset.csv does not look like the original dataset. "
            f"Columns found: {list(df.columns)[:8]} ... Re-upload the original CSV to the data folder of your repo."
        )
        st.stop()
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(df["Arrival Delay in Minutes"].median())
    return df


@st.cache_resource(show_spinner="Warming up the engines (first launch trains the model, about a minute)...")
def load_artifacts():
    needed = ["XG_Boost_model.pkl", "scaler.pkl", "label_encoders.pkl"]
    if not all((MODEL_DIR / n).exists() for n in needed):
        import train_model
        train_model.main()
    model = joblib.load(MODEL_DIR / "XG_Boost_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    encoders = joblib.load(MODEL_DIR / "label_encoders.pkl")
    mp = MODEL_DIR / "metrics.json"
    metrics = json.loads(mp.read_text()) if mp.exists() else None
    return model, scaler, encoders, metrics


def _frame(passenger: dict, encoders) -> pd.DataFrame:
    row = dict(passenger)
    for col in ["Gender", "Customer Type", "Type of Travel", "Class"]:
        row[col] = int(encoders[col].transform([row[col]])[0])
    return pd.DataFrame([row])[FEATURES]


def predict(passenger: dict) -> float:
    """Probability (0-1) that the passenger is satisfied."""
    model, scaler, encoders, _ = load_artifacts()
    X = scaler.transform(_frame(passenger, encoders))
    sat_idx = list(model.classes_).index(int(encoders["satisfaction"].transform(["satisfied"])[0]))
    return float(model.predict_proba(X)[0][sat_idx])


def best_levers(passenger: dict, top: int = 3):
    """What-if: set each service rating to 5 and see how much satisfaction probability moves."""
    base = predict(passenger)
    gains = []
    for col in SERVICE_COLS:
        if passenger[col] >= 5:
            continue
        trial = dict(passenger); trial[col] = 5
        gains.append((col, predict(trial) - base))
    gains.sort(key=lambda g: g[1], reverse=True)
    return [g for g in gains[:top] if g[1] > 0.005]
