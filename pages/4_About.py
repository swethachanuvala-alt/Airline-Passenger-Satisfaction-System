import streamlit as st
from utils import setup, html, gate_header, photo

setup("About")
gate_header("4", "About", "The pipeline behind the boarding pass.")

left, right = st.columns([1.2, 1])
with left:
    st.markdown("""
#### The pipeline
1. **Clean:** the one column with gaps, Arrival Delay, is filled with its median.
2. **Encode:** gender, loyalty, trip purpose and cabin class become numbers with label encoders.
3. **Split:** 80% training, 20% test, stratified so both keep the same mix of satisfied passengers.
4. **Scale:** a standard scaler is fitted on the training data only.
5. **Balance:** SMOTE creates extra examples of the smaller class in the training set.
6. **Train and tune:** five models were tuned with randomized search; XGBoost won.

#### The data
49,896 airline passengers with 22 inputs: who they are, what trip they took, 14 service ratings
from 0 to 5, and their delays. The target is whether they were satisfied or neutral/dissatisfied.

#### A note on the inputs
The notebook's row index and passenger id columns were left in the features. They say nothing about
a passenger, so this app retrains the same pipeline without them and only asks for real inputs.
""")
with right:
    photo("terminal_gate.jpg", "The images in this app are illustrations. Replace any file in assets/images with a real photo of the same name and it will be used automatically.")
    html("""<div class="card"><h4>Built with</h4>
    <p>Python, pandas, scikit-learn, imbalanced-learn, XGBoost, Plotly and Streamlit.</p></div>""")
