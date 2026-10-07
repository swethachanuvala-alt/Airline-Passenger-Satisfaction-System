import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils import setup, gate_header, load_artifacts, FEATURES, NAVY, TEAL, YELLOW, photo

setup("Control Tower")
gate_header("3", "Control Tower", "How XGBoost was chosen, how well it performs, and what it pays attention to.")

model, scaler, encoders, metrics = load_artifacts()

# Tuned results copied from the notebook (GGST_10.ipynb)
nb = pd.DataFrame({
    "Model": ["XGBoost", "Random Forest", "Decision Tree", "KNN", "Logistic Regression"],
    "Accuracy": [0.9651, 0.9608, 0.9473, 0.9296, 0.8774],
    "Precision": [0.9688, 0.9667, 0.9415, 0.9287, 0.8539],
    "Recall": [0.9506, 0.9425, 0.9372, 0.9080, 0.8668],
    "F1 score": [0.9596, 0.9545, 0.9394, 0.9182, 0.8603],
    "ROC-AUC": [0.9953, 0.9935, 0.9671, 0.9766, 0.9296],
})

st.markdown("### Model line-up (tuned, from the notebook)")
c1, c2 = st.columns([1.1, 1])
with c1:
    st.dataframe(nb.style.format({k: "{:.4f}" for k in nb.columns[1:]}).highlight_max(
        subset=nb.columns[1:], color="#FFE9A8"), hide_index=True, use_container_width=True)
    st.caption("Winner on every metric. Five-fold cross-validation accuracy for tuned XGBoost was 0.9606 (std 0.0046).")
with c2:
    fig = px.bar(nb, x="Accuracy", y="Model", orientation="h", color_discrete_sequence=[TEAL], range_x=[0.8, 1.0])
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff", margin=dict(l=10, r=10, t=10, b=10),
                      font=dict(family="Barlow, sans-serif", color=NAVY))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### The model running in this app")
if metrics:
    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    b.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
    c.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
    d.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    cm = metrics["confusion"]
else:
    cm = [[5499, 133], [215, 4133]]
    st.caption("Showing the notebook's confusion matrix.")

left, right = st.columns(2)
with left:
    z = [[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]]
    fig = go.Figure(go.Heatmap(z=z, x=["Predicted: not satisfied", "Predicted: satisfied"],
                               y=["Actual: not satisfied", "Actual: satisfied"], text=z,
                               texttemplate="%{text}", colorscale=[[0, "#FFFFFF"], [1, TEAL]], showscale=False))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Confusion matrix on the held-out 20%", height=380, paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Barlow, sans-serif", color=NAVY, size=14), margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
with right:
    imp = pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_}).sort_values("Importance").tail(12)
    fig = px.bar(imp, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[YELLOW],
                 title="What the model pays attention to")
    fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff",
                      font=dict(family="Barlow, sans-serif", color=NAVY), margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

photo("runway_night.jpg")
