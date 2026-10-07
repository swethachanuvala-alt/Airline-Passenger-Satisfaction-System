import pandas as pd
import plotly.express as px
import streamlit as st
from utils import setup, gate_header, load_data, photo, SERVICE_COLS, NAVY, TEAL, RED, YELLOW

setup("Flight Insights")
gate_header("2", "Flight Insights", "What separates satisfied passengers from the rest? Filter the survey and see.")

df = load_data()
COLORS = {"satisfied": TEAL, "neutral or dissatisfied": RED}

f1, f2, f3 = st.columns(3)
classes = f1.multiselect("Cabin class", sorted(df["Class"].unique()), default=sorted(df["Class"].unique()))
trips = f2.multiselect("Purpose of trip", sorted(df["Type of Travel"].unique()), default=sorted(df["Type of Travel"].unique()))
loyal = f3.multiselect("Loyalty", sorted(df["Customer Type"].unique()), default=sorted(df["Customer Type"].unique()))
d = df[df["Class"].isin(classes) & df["Type of Travel"].isin(trips) & df["Customer Type"].isin(loyal)]
if d.empty:
    st.warning("No passengers match these filters.")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Passengers", f"{len(d):,}")
m2.metric("Satisfied", f"{(d['satisfaction'] == 'satisfied').mean() * 100:.1f}%")
m3.metric("Average age", f"{d['Age'].mean():.0f}")
m4.metric("Average distance", f"{d['Flight Distance'].mean():,.0f} mi")


def style(fig, h=380):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff",
                      font=dict(family="Barlow, sans-serif", color=NAVY), margin=dict(l=10, r=10, t=50, b=10),
                      legend_title_text="")
    return fig


left, right = st.columns(2)
with left:
    g = d.groupby(["Class", "satisfaction"]).size().reset_index(name="n")
    g["pct"] = g["n"] / g.groupby("Class")["n"].transform("sum") * 100
    fig = px.bar(g, x="Class", y="pct", color="satisfaction", color_discrete_map=COLORS,
                 title="Satisfaction by cabin class (%)", labels={"pct": "% of passengers"})
    st.plotly_chart(style(fig), use_container_width=True)
with right:
    g = d.groupby(["Type of Travel", "satisfaction"]).size().reset_index(name="n")
    g["pct"] = g["n"] / g.groupby("Type of Travel")["n"].transform("sum") * 100
    fig = px.bar(g, x="Type of Travel", y="pct", color="satisfaction", color_discrete_map=COLORS,
                 title="Satisfaction by purpose of trip (%)", labels={"pct": "% of passengers"})
    st.plotly_chart(style(fig), use_container_width=True)

photo("window_wing.jpg")

means = d.groupby("satisfaction")[SERVICE_COLS].mean().T.reset_index().melt(
    id_vars="index", var_name="satisfaction", value_name="rating")
fig = px.bar(means, y="index", x="rating", color="satisfaction", barmode="group", orientation="h",
             color_discrete_map=COLORS, title="Average rating of each service (0 to 5)",
             labels={"index": "", "rating": "Average rating"})
fig.update_yaxes(autorange="reversed")
st.plotly_chart(style(fig, 560), use_container_width=True)

left, right = st.columns(2)
with left:
    fig = px.histogram(d, x="Age", color="satisfaction", nbins=30, barmode="overlay", opacity=.75,
                       color_discrete_map=COLORS, title="Age of passengers")
    st.plotly_chart(style(fig), use_container_width=True)
with right:
    b = d.copy()
    b["Delay band"] = pd.cut(b["Departure Delay in Minutes"], [-1, 0, 15, 60, 180, 5000],
                             labels=["On time", "1-15 min", "16-60 min", "1-3 hours", "3+ hours"])
    g = b.groupby("Delay band", observed=True)["satisfaction"].apply(lambda s: (s == "satisfied").mean() * 100).reset_index(name="pct")
    fig = px.bar(g, x="Delay band", y="pct", title="Satisfied passengers by departure delay (%)",
                 labels={"pct": "% satisfied"}, color_discrete_sequence=[YELLOW])
    st.plotly_chart(style(fig), use_container_width=True)
