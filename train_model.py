"""
Reproduces the notebook (GGST_10.ipynb) pipeline and saves the artifacts the app needs.

Run once:   python train_model.py

Pipeline (same as notebook): median-fill Arrival Delay -> LabelEncoder on categoricals ->
80/20 stratified split (random_state=42) -> StandardScaler -> SMOTE on train -> XGBoost
(tuned params found by the notebook's RandomizedSearchCV).

One deliberate change: the notebook's row-index column ("Unnamed: 0") and "id" were left in
the features. They carry no real information about a passenger, so they are dropped here and
the app only asks for genuine passenger / flight inputs.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "airline_satisfaction_dataset.csv"
MODELS = ROOT / "models"

CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
BEST_PARAMS = dict(subsample=1.0, n_estimators=300, max_depth=7,
                   learning_rate=0.1, colsample_bytree=0.8)


def main():
    MODELS.mkdir(exist_ok=True)
    df = pd.read_csv(DATA, encoding="utf-8-sig", skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.drop(columns=[c for c in ["Unnamed: 0", "id"] if c in df.columns])
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(
        df["Arrival Delay in Minutes"].median())

    encoders = {}
    for col in CAT_COLS + ["satisfaction"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X, y = df.drop("satisfaction", axis=1), df["satisfaction"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    X_res, y_res = SMOTE(random_state=42).fit_resample(X_tr_s, y_tr)

    model = XGBClassifier(random_state=42, eval_metric="logloss", **BEST_PARAMS)
    model.fit(X_res, y_res)

    pred = model.predict(X_te_s)
    prob = model.predict_proba(X_te_s)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    metrics = dict(
        accuracy=accuracy_score(y_te, pred), precision=precision_score(y_te, pred),
        recall=recall_score(y_te, pred), f1=f1_score(y_te, pred),
        roc_auc=roc_auc_score(y_te, prob),
        confusion=[[int(tn), int(fp)], [int(fn), int(tp)]],
        features=list(X.columns), n_train=int(len(X_tr)), n_test=int(len(X_te)),
    )

    joblib.dump(model, MODELS / "XG_Boost_model.pkl")
    joblib.dump(scaler, MODELS / "scaler.pkl")
    joblib.dump(encoders, MODELS / "label_encoders.pkl")
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model, scaler, encoders. Test accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
