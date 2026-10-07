import streamlit as st
from utils import setup, html, gate_header, predict, best_levers, load_artifacts

setup("Check In")
gate_header("1", "Check In", "Describe the passenger and the trip, rate the service, and get a boarding pass with the predicted verdict.")

load_artifacts()  # warm the model before the form is submitted

with st.form("checkin"):
    st.markdown("#### Passenger and trip")
    c1, c2, c3, c4 = st.columns(4)
    gender = c1.selectbox("Gender", ["Female", "Male"])
    age = c2.number_input("Age", 5, 90, 35)
    ctype = c3.selectbox("Loyalty", ["Loyal Customer", "disloyal Customer"],
                         format_func=lambda v: "Loyal customer" if v.startswith("Loyal") else "Disloyal customer")
    travel = c4.selectbox("Purpose of trip", ["Business travel", "Personal Travel"],
                          format_func=lambda v: "Business" if v.startswith("Business") else "Personal")
    c5, c6, c7, c8 = st.columns(4)
    cls = c5.selectbox("Cabin class", ["Business", "Eco Plus", "Eco"])
    dist = c6.number_input("Flight distance (miles)", 30, 5000, 1200, step=50)
    dep = c7.number_input("Departure delay (min)", 0, 1700, 0, step=5)
    arr = c8.number_input("Arrival delay (min)", 0, 1700, 0, step=5)

    st.markdown("#### Rate the service (0 = not applicable, 1 = poor, 5 = excellent)")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.caption("Booking and airport")
        wifi = st.slider("Inflight wifi", 0, 5, 3)
        timing = st.slider("Departure/arrival time convenience", 0, 5, 3)
        booking = st.slider("Ease of online booking", 0, 5, 3)
        gate = st.slider("Gate location", 1, 5, 3)
        checkin = st.slider("Check-in service", 0, 5, 3)
    with g2:
        st.caption("Boarding and seat")
        board = st.slider("Online boarding", 0, 5, 3)
        seat = st.slider("Seat comfort", 0, 5, 3)
        leg = st.slider("Leg room", 0, 5, 3)
        bag = st.slider("Baggage handling", 1, 5, 3)
        clean = st.slider("Cleanliness", 0, 5, 3)
    with g3:
        st.caption("On board")
        food = st.slider("Food and drink", 0, 5, 3)
        ent = st.slider("Inflight entertainment", 0, 5, 3)
        onb = st.slider("On-board service", 0, 5, 3)
        infl = st.slider("Inflight service", 0, 5, 3)
    go = st.form_submit_button("Issue boarding pass", type="primary", use_container_width=True)

if go:
    p = {
        "Gender": gender, "Customer Type": ctype, "Age": age, "Type of Travel": travel,
        "Class": cls, "Flight Distance": dist, "Inflight wifi service": wifi,
        "Departure/Arrival time convenient": timing, "Ease of Online booking": booking,
        "Gate location": gate, "Food and drink": food, "Online boarding": board,
        "Seat comfort": seat, "Inflight entertainment": ent, "On-board service": onb,
        "Leg room service": leg, "Baggage handling": bag, "Checkin service": checkin,
        "Inflight service": infl, "Cleanliness": clean,
        "Departure Delay in Minutes": dep, "Arrival Delay in Minutes": arr,
    }
    prob = predict(p)
    ok = prob >= 0.5
    pct = prob * 100
    stamp = '<div class="stamp ok">SATISFIED</div>' if ok else '<div class="stamp bad">NOT SATISFIED</div>'
    note = ("This passenger is likely to step off the plane happy." if ok
            else "This passenger is likely to leave neutral or dissatisfied.")
    cabin = {"Eco Plus": "Eco Plus"}.get(cls, cls)
    html(f"""
    <div class="pass">
    <div class="pass-main">
    <div class="pass-top"><span class="pass-airline">✈ SkyPulse Air</span><span class="pass-type">BOARDING PASS</span></div>
    <div class="pass-grid">
    <div><div class="l">Passenger</div><div class="v">{gender}, {age}</div></div>
    <div><div class="l">Cabin</div><div class="v">{cabin}</div></div>
    <div><div class="l">Trip</div><div class="v">{'Business' if travel.startswith('Business') else 'Personal'}</div></div>
    <div><div class="l">Distance</div><div class="v">{dist:,} mi</div></div>
    <div><div class="l">Loyalty</div><div class="v">{'Loyal' if ctype.startswith('Loyal') else 'Disloyal'}</div></div>
    <div><div class="l">Departure delay</div><div class="v">{dep} min</div></div>
    <div><div class="l">Arrival delay</div><div class="v">{arr} min</div></div>
    <div><div class="l">Average service</div><div class="v">{sum([wifi,timing,booking,gate,food,board,seat,ent,onb,leg,bag,checkin,infl,clean])/14:.1f} / 5</div></div>
    </div>
    <div class="pass-verdict">{stamp}<div class="pass-note">{note}</div></div>
    </div>
    <div class="pass-stub">
    <div><div class="l">Satisfaction probability</div><div class="big">{pct:.0f}%</div>
    <div class="bar-track"><div class="bar-fill" style="width:{pct:.0f}%"></div></div></div>
    <div class="barcode"></div>
    </div>
    </div>
    """)

    levers = best_levers(p)
    if not ok and levers:
        st.markdown("#### What would help most")
        st.caption("Each line shows the change in satisfaction probability if that rating became a 5, all else equal.")
        for col, gain in levers:
            st.markdown(f"- **{col}**: +{gain * 100:.0f} percentage points")
    elif ok:
        st.caption("Estimate from an XGBoost model trained on past survey responses; treat it as a guide, not a certainty.")
