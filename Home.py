import streamlit as st
from utils import setup, html, img_b64, photo, load_data

setup("Terminal")

df = load_data()
n = len(df)
sat = (df["satisfaction"] == "satisfied").mean() * 100

hero_bg = img_b64("hero_takeoff.jpg")
bg = (f"linear-gradient(90deg, rgba(14,34,56,.90) 0%, rgba(14,34,56,.45) 52%, rgba(14,34,56,0) 100%), "
      f"url(data:image/jpeg;base64,{hero_bg})") if hero_bg else "#0E2238"
html(f"""
<div class="hero" style="background-image:{bg};">
<div class="hero-copy">
<div class="hero-sign">Now boarding: insight</div>
<h1>Will your passengers land happy?</h1>
<p>SkyPulse reads a passenger's trip, their ratings of the service and the delays they faced,
then tells you whether they walked off the plane satisfied. Built on {n:,} real airline survey responses.</p>
</div>
</div>
""")

st.write("")
c1, c2, c3, _ = st.columns([1.1, 1.1, 1.1, 2.2])
c1.page_link("pages/1_Check_In.py", label="Check a passenger in", icon="🎫")
c2.page_link("pages/2_Flight_Insights.py", label="Flight insights", icon="📊")
c3.page_link("pages/3_Control_Tower.py", label="Control tower", icon="🧠")

st.markdown('<div class="section-title">Today\'s departures</div>', unsafe_allow_html=True)
html(f"""
<div class="board">
<div class="board-head"><span>Flight record</span><span>Value</span><span>Status</span></div>
<div class="board-row"><span>Surveyed</span><span>{n:,}</span><span class="st">On record</span></div>
<div class="board-row"><span>Satisfied</span><span>{sat:.1f}%</span><span class="st">Boarding</span></div>
<div class="board-row"><span>Not satisfied</span><span>{100 - sat:.1f}%</span><span class="st warn">Delayed</span></div>
<div class="board-row"><span>Model</span><span>XGBoost</span><span class="st">96.5% accurate</span></div>
</div>
""")

st.markdown('<div class="section-title">Explore the terminal</div>', unsafe_allow_html=True)
a, b, c = st.columns(3)
with a:
    photo("terminal_gate.jpg")
    html("""<div class="card"><h4>Gate 1: Check In</h4>
    <p>Describe a passenger and trip. The model issues a boarding pass with the satisfaction
    verdict and the service fixes that would help most.</p></div>""")
with b:
    photo("window_wing.jpg")
    html("""<div class="card"><h4>Gate 2: Flight Insights</h4>
    <p>Filter the survey by class, trip purpose and loyalty to see what separates happy
    passengers from unhappy ones.</p></div>""")
with c:
    photo("runway_night.jpg")
    html("""<div class="card"><h4>Gate 3: Control Tower</h4>
    <p>How XGBoost compared with four other models, its confusion matrix and which
    features drive its decisions.</p></div>""")
