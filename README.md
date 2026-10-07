# SkyPulse: Airline Passenger Satisfaction Terminal

Multipage Streamlit app (airport theme) around an XGBoost model that predicts whether an airline passenger is satisfied.

## Run locally (Windows cmd, from this folder)

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    python train_model.py
    streamlit run Home.py

`train_model.py` rebuilds the notebook pipeline and saves the model, scaler and encoders into `models/`.
Run it once; commit the generated files so deployment is instant. (If you skip it, the app trains the
model itself on first launch, which takes about a minute.)

## Push to GitHub

    git init
    git add .
    git commit -m "SkyPulse airline satisfaction app"
    git branch -M main
    git remote add origin https://github.com/<your-username>/<repo-name>.git
    git push -u origin main

## Deploy on Streamlit Community Cloud

1. Go to share.streamlit.io and sign in with GitHub.
2. New app, pick your repo, branch `main`, main file path `Home.py`.
3. Deploy.

## Folder layout

    Home.py                 landing page
    pages/                  Check In, Flight Insights, Control Tower, About
    utils.py                theme, model loading, prediction
    train_model.py          retrains and saves the model
    data/                   dataset
    models/                 saved model files (after training)
    assets/images/          airport images (replace with real photos of the same name if you like)
