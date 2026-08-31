"""
IEEE-CIS Fraud Detection -- Streamlit demo.

Draws a random batch of rows from your held-out 1000-row sample, then
SIMULATES them arriving one at a time: each transaction is shown in the
UI, scored by the saved LightGBM booster, and appended to a live results
table. The simulation only runs when the user clicks "Run Simulation".
Once it finishes, a full evaluation dashboard (metrics, curves, feature
importance, downloadable table) is shown below.

Run:
    streamlit run app.py

Expected files in the working directory (or set custom paths in the sidebar):
    - sample_1000.csv           raw 1000-row holdout sample (1000, 434), incl. isFraud
    - pipeline_artifacts.pkl    produced by fit_preprocessing() on Kaggle
    - lgb_gbdt.txt              your saved LightGBM booster
"""
import time
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import lightgbm as lgb
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_score, recall_score, f1_score, roc_curve, precision_recall_curve,
)

from fraud_pipeline import transform_new

st.set_page_config(page_title="Fraud Detection Demo", layout="wide")

DEFAULT_SAMPLE_PATH = "../data/sample_1000.csv"
DEFAULT_ARTIFACTS_PATH = "../models/pipeline_artifacts.pkl"
DEFAULT_MODEL_PATH = "../models/lgb_gbdt.txt"


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_resource
def load_model(path):
    return lgb.Booster(model_file=path)


@st.cache_resource
def load_artifacts(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_sample(path):
    return pd.read_csv(path)


# --------------------------------------------------------------------------
# Sidebar: file locations + run controls
# --------------------------------------------------------------------------
# st.sidebar.title("⚙️ Configuration")

# sample_file = st.sidebar.text_input("Sample CSV path", DEFAULT_SAMPLE_PATH)
# artifacts_file = st.sidebar.text_input("Pipeline artifacts (.pkl)", DEFAULT_ARTIFACTS_PATH)
# model_file = st.sidebar.text_input("LightGBM model (.txt)", DEFAULT_MODEL_PATH)

# st.sidebar.markdown("---")
n_rows = st.sidebar.slider("Transactions to simulate", min_value=10, max_value=10000, value=100, step=10)
seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)
speed = st.sidebar.select_slider(
    "Simulation speed",
    options=["Slow", "Normal", "Fast", "Instant"],
    value="Fast",
)
threshold=0.015
# threshold = st.sidebar.slider("Decision threshold", min_value=0.0, max_value=1.0, value=0.01, step=0.01)
# threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.50, 0.01)
if 'results' in st.session_state and 'isFraud' in st.session_state['results']['batch'].columns:
    _y = st.session_state['results']['batch']['isFraud'].values
    _p = st.session_state['results']['proba']
    if len(np.unique(_y)) > 1:
        _prec, _rec, _thr = precision_recall_curve(_y, _p)
        _f1 = np.where((_prec + _rec) > 0, 2 * _prec * _rec / (_prec + _rec + 1e-12), 0)
        _best_idx = np.argmax(_f1[:-1]) if len(_thr) > 0 else None
        if _best_idx is not None and len(_thr) > 0:
            st.sidebar.caption(
                f"💡 Threshold maximizing F1 on this batch: **{_thr[_best_idx]:.2f}** "
                f"(precision {_prec[_best_idx]:.2f}, recall {_rec[_best_idx]:.2f})"
            )
run_btn = st.sidebar.button("▶️ Run Simulation", type="primary", use_container_width=True)

SPEED_DELAY = {"Slow": 0.35, "Normal": 0.15, "Fast": 0.04, "Instant": 0.0}

st.title("💳 IEEE-CIS Fraud Detection — Live Demo")
st.caption(
    "Simulates a stream of transactions from a 1,000-row holdout sample never seen "
    "during training. Each transaction runs through the full preprocessing pipeline "
    "(missing/variance filtering → V-feature PCA → feature engineering → encoding) "
    "and is scored by the tuned LightGBM GBDT model in real time."
)

# --------------------------------------------------------------------------
# Load resources
# --------------------------------------------------------------------------
try:
    model = load_model(DEFAULT_MODEL_PATH)
    artifacts = load_artifacts(DEFAULT_ARTIFACTS_PATH)
    sample_df = load_sample(DEFAULT_SAMPLE_PATH)
except FileNotFoundError as e:
    st.error(
        f"Couldn't find one of the required files: `{e.filename}`.\n\n"
        "Make sure `sample_1000.csv`, `pipeline_artifacts.pkl` and `lgb_gbdt.txt` are "
        "either in this app's working directory, or update the paths in the sidebar."
    )
    st.stop()

st.sidebar.success(f"Loaded {len(sample_df):,} sample rows · {len(artifacts['FEATURES'])} features")
if 'isFraud' not in sample_df.columns:
    st.sidebar.warning("Sample has no `isFraud` column — evaluation metrics will be skipped.")


# --------------------------------------------------------------------------
# Simulation (only runs on button click)
# --------------------------------------------------------------------------
def run_simulation(batch, X, delay):
    progress = st.progress(0.0, text="Starting simulation…")
    current_box = st.empty()
    table_box = st.empty()

    rows = []
    n = len(batch)
    for i in range(n):
        txn_id = batch['TransactionID'].iloc[i] if 'TransactionID' in batch.columns else i
        amt = batch['TransactionAmt'].iloc[i] if 'TransactionAmt' in batch.columns else None

        with current_box.container():
            st.markdown(f"**Processing transaction `{txn_id}`**" +
                        (f" — ${amt:,.2f}" if amt is not None else ""))
            st.caption("Running: drop → V-PCA → feature engineering → encoding → predict…")

        proba = float(model.predict(X.iloc[[i]])[0])
        pred_label = "🚨 Fraud" if proba >= threshold else "✅ Legit"

        row = {"TransactionID": txn_id, "fraud_probability": round(proba, 4), "prediction": pred_label}
        if 'isFraud' in batch.columns:
            actual = int(batch['isFraud'].iloc[i])
            row["actual"] = "Fraud" if actual == 1 else "Legit"
        rows.append(row)

        with current_box.container():
            flag = "🚨" if proba >= threshold else "✅"
            st.markdown(f"**Transaction `{txn_id}`** → probability **{proba:.1%}** {flag}")

        table_box.dataframe(
            pd.DataFrame(rows[::-1]),  # newest on top
            use_container_width=True, height=280,
        )
        progress.progress((i + 1) / n, text=f"Scored {i + 1} / {n} transactions")

        if delay > 0:
            time.sleep(delay)

    progress.empty()
    current_box.empty()
    return rows


if run_btn:
    batch = sample_df.sample(n=min(n_rows, len(sample_df)), random_state=int(seed)).reset_index(drop=True)
    with st.spinner("Preparing features for this batch…"):
        X, transformed = transform_new(batch, artifacts)

    st.subheader("🔴 Live simulation")
    run_simulation(batch, X, SPEED_DELAY[speed])
    proba_all = model.predict(X)

    st.session_state['results'] = dict(batch=batch, X=X, proba=proba_all)
    st.success("Simulation complete — full results below.")

if 'results' not in st.session_state:
    st.info("Set your options in the sidebar and click **▶️ Run Simulation** to begin.")
    st.stop()

batch = st.session_state['results']['batch']
X = st.session_state['results']['X']
proba = st.session_state['results']['proba']
pred = (proba >= threshold).astype(int)
has_labels = 'isFraud' in batch.columns

st.markdown("---")

# --------------------------------------------------------------------------
# Top-line metrics
# --------------------------------------------------------------------------
st.subheader("📊 Batch summary")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows scored", f"{len(batch):,}")
c2.metric("Predicted fraud", f"{pred.sum():,}", f"{pred.mean():.1%} of batch")
if has_labels:
    y_true = batch['isFraud'].values
    c3.metric("Actual fraud", f"{int(y_true.sum()):,}", f"{y_true.mean():.1%} of batch")
    try:
        c4.metric("ROC-AUC", f"{roc_auc_score(y_true, proba):.3f}")
        c5.metric("PR-AUC", f"{average_precision_score(y_true, proba):.3f}")
    except ValueError:
        c4.metric("ROC-AUC", "n/a")
        c5.metric("PR-AUC", "n/a")
else:
    c3.metric("Actual fraud", "n/a")
    c4.metric("ROC-AUC", "n/a")
    c5.metric("PR-AUC", "n/a")

# --------------------------------------------------------------------------
# Evaluation plots (only if ground truth available)
# --------------------------------------------------------------------------
if has_labels:
    st.subheader("🎯 Evaluation")
    col1, col2, col3 = st.columns(3)

    with col1:
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Legit", "Fraud"], y=["Legit", "Fraud"],
        )
        fig_cm.update_layout(title="Confusion Matrix", height=350)
        st.plotly_chart(fig_cm, use_container_width=True)
        st.write(
            f"**True Positives:** {tp}  ·  **False Negatives:** {fn}  \n"
            f"**True Negatives:** {tn}  ·  **False Positives:** {fp}  \n"
            f"**Precision:** {precision_score(y_true, pred, zero_division=0):.3f}  \n"
            f"**Recall:** {recall_score(y_true, pred, zero_division=0):.3f}  \n"
            f"**F1:** {f1_score(y_true, pred, zero_division=0):.3f}"
        )

    with col2:
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, proba)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                          line=dict(dash="dash", color="gray"), name="Chance"))
            fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", height=350)
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info("Only one class present in this batch — ROC curve not meaningful.")

    with col3:
        if len(np.unique(y_true)) > 1:
            prec, rec, _ = precision_recall_curve(y_true, proba)
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR"))
            fig_pr.update_layout(title="Precision-Recall Curve", xaxis_title="Recall",
                                  yaxis_title="Precision", height=350)
            st.plotly_chart(fig_pr, use_container_width=True)
        else:
            st.info("Only one class present in this batch — PR curve not meaningful.")

# --------------------------------------------------------------------------
# Probability distribution
# --------------------------------------------------------------------------
st.subheader("📈 Predicted fraud probability distribution")
log_y = st.checkbox(
    "Log scale (y-axis)", value=True,
    help="With few fraud cases in the batch, a linear scale hides them under the "
         "legit pile near 0. Log scale makes rare high-probability bars visible."
)
hist_df = pd.DataFrame({"probability": proba})
if has_labels:
    hist_df["actual"] = np.where(batch['isFraud'].values == 1, "Fraud", "Legit")
    fig_hist = px.histogram(hist_df, x="probability", color="actual", nbins=40, barmode="overlay",
                             color_discrete_map={"Fraud": "#EF553B", "Legit": "#636EFA"},
                             log_y=log_y)
else:
    fig_hist = px.histogram(hist_df, x="probability", nbins=40, log_y=log_y)
fig_hist.add_vline(x=threshold, line_dash="dash", line_color="black",
                    annotation_text=f"threshold={threshold:.2f}")
fig_hist.update_layout(height=350)
st.plotly_chart(fig_hist, use_container_width=True)

if has_labels:
    st.caption(
        f"Fraud probabilities in this batch range from "
        f"{proba[batch['isFraud'].values == 1].min():.3f} to "
        f"{proba[batch['isFraud'].values == 1].max():.3f} "
        f"(median {np.median(proba[batch['isFraud'].values == 1]):.3f}). "
        f"Legit probabilities range from {proba[batch['isFraud'].values == 0].min():.3f} to "
        f"{proba[batch['isFraud'].values == 0].max():.3f}."
    )

# --------------------------------------------------------------------------
# Feature importance
# --------------------------------------------------------------------------
st.subheader("🔍 Top model features")
imp = pd.DataFrame({
    "feature": model.feature_name(),
    "importance": model.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=False).head(20)
fig_imp = px.bar(imp.sort_values("importance"), x="importance", y="feature", orientation="h")
fig_imp.update_layout(height=500)
st.plotly_chart(fig_imp, use_container_width=True)

# --------------------------------------------------------------------------
# Row-level table
# --------------------------------------------------------------------------
st.subheader("🧾 Row-level predictions")
display_cols = {}
if 'TransactionID' in batch.columns:
    display_cols['TransactionID'] = batch['TransactionID'].values
if 'TransactionAmt' in batch.columns:
    display_cols['TransactionAmt'] = batch['TransactionAmt'].values
display_cols['fraud_probability'] = np.round(proba, 4)
display_cols['prediction'] = np.where(pred == 1, "Fraud", "Legit")
if has_labels:
    display_cols['actual'] = np.where(batch['isFraud'].values == 1, "Fraud", "Legit")
    display_cols['correct'] = pred == batch['isFraud'].values

result_table = pd.DataFrame(display_cols).sort_values("fraud_probability", ascending=False)
st.dataframe(result_table, use_container_width=True, height=450)

st.download_button(
    "⬇️ Download predictions as CSV",
    result_table.to_csv(index=False).encode("utf-8"),
    file_name="fraud_predictions.csv",
    mime="text/csv",
)








