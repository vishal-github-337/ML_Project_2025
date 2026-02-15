import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
)

# -------------------------
# Config & helpers
# -------------------------
st.set_page_config(page_title="Heart Disease Risk Predictor", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_model(path):
    if not path:
        return None
    return joblib.load(path)

MODEL_FILENAME_MAP = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
    "Gradient Boosting": "gradient_boosting.joblib"
}

# -------------------------
# CSS / theme
# -------------------------
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#f7fbfc,#eef6f9); }
    .header-card {
        background: linear-gradient(90deg,#ffefef,#eef7ff);
        padding:16px;
        border-radius:12px;
        box-shadow: 0 6px 20px rgba(8,20,40,0.04);
    }
    .metric {
        background: white;
        padding:12px;
        border-radius:8px;
        box-shadow: 0 4px 12px rgba(8,20,40,0.04);
        text-align:center;
    }
    .small-muted { color:#6b7280; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Header
# -------------------------
with st.container():
    st.markdown(
        f"""
        <div class="header-card">
            <h1 style="margin:0;">🫀 Heart Disease Risk Predictor</h1>
            <div class="small-muted">Clinical screening-focused. RECALL (sensitivity) prioritized to reduce missed cases.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Test Data)", type=["csv"])
selected_model_name = st.sidebar.selectbox("Select model", list(MODEL_FILENAME_MAP.keys()))
show_raw = st.sidebar.checkbox("Show raw data", value=False)
threshold = st.sidebar.slider("Probability threshold for positive label", 0.0, 1.0, 0.5, 0.01)
st.sidebar.markdown("Models with higher recall are preferred for clinical screening.")

# -------------------------
# Main
# -------------------------
if uploaded_file is None:
    st.info("Upload a CSV with features + `Heart_Risk` target (0/1). Download `test.csv` from repo if needed.")
    st.stop()

# read file
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

if show_raw:
    st.write("### Dataset preview")
    st.dataframe(df.head(10))

target_col = "Heart_Risk"
if target_col not in df.columns:
    st.error(f"CSV must include the target column `{target_col}`.")
    st.stop()

# coerce floats that are actually ints
float_cols = df.select_dtypes(include=["float"]).columns
if len(float_cols) > 0:
    try:
        df[float_cols] = df[float_cols].apply(lambda s: np.where(np.isclose(s, np.round(s)), np.round(s).astype(int), s))
    except Exception:
        pass

X_test = df.drop(columns=[target_col])
y_test = df[target_col]

# load model
model_filename = MODEL_FILENAME_MAP.get(selected_model_name)
model_path = f"model/{model_filename}"
try:
    model = load_model(model_path)
except FileNotFoundError:
    st.error(f"Model file not found: {model_path}. Run `model_training.py` to create models.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# predict
try:
    # If model has predict_proba, use probabilities and threshold
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        y_pred = (probs >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        probs = None
except Exception as e:
    st.error(f"Prediction error: {e}")
    st.stop()

# metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred) if len(np.unique(y_test)) > 1 else 0.0
auc = roc_auc_score(y_test, probs) if probs is not None and len(np.unique(y_test)) > 1 else None

# confusion matrix & specificity
cm = confusion_matrix(y_test, y_pred)
if cm.size == 4:
    tn, fp, fn, tp = cm.ravel()
else:
    tn = fp = fn = tp = 0
specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0

# Top metrics - emphasize recall
st.markdown("### Performance Summary")
c1, c2, c3, c4, c5 = st.columns([1.6, 1, 1, 1, 1])
with c1:
    st.markdown(f"""
        <div class="metric">
            <h4 style="color:#e63946; margin:0;">🔴 RECALL</h4>
            <h2 style="color:#e63946; margin:0;">{rec:.4f}</h2>
            <div class="small-muted">Sensitivity — detected positives / actual positives</div>
        </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
        <div class="metric">
            <h4 style="color:#0077b6; margin:0;">Specificity</h4>
            <h3 style="margin:0;color:#0077b6;">{specificity:.4f}</h3>
            <div class="small-muted">True negative rate</div>
        </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown(f"""
        <div class="metric">
            <h4 style="margin:0;">Precision</h4>
            <h3 style="margin:0;">{prec:.4f}</h3>
            <div class="small-muted">Positive predictive value</div>
        </div>
    """, unsafe_allow_html=True)
with c4:
    st.markdown(f"""
        <div class="metric">
            <h4 style="margin:0;">F1</h4>
            <h3 style="margin:0;">{f1:.4f}</h3>
            <div class="small-muted">Harmonic mean of precision & recall</div>
        </div>
    """, unsafe_allow_html=True)
with c5:
    auc_text = f"{auc:.4f}" if auc is not None else "N/A"
    st.markdown(f"""
        <div class="metric">
            <h4 style="margin:0;">AUC</h4>
            <h3 style="margin:0;">{auc_text}</h3>
            <div class="small-muted">ROC AUC (if probabilities available)</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Confusion matrix + clinical notes
left, right = st.columns([2, 1])
with left:
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlGn_r", ax=ax,
                xticklabels=["Healthy (0)", "At Risk (1)"],
                yticklabels=["Healthy (0)", "At Risk (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.write("### Clinical interpretation")
    st.markdown(f"- **True Positives (TP):** {tp}")
    st.markdown(f"- **False Negatives (FN):** {fn}  ⚠️ *Missed cases — critical*")
    st.markdown(f"- **False Positives (FP):** {fp}")
    st.markdown(f"- **True Negatives (TN):** {tn}")
    st.markdown(f"**Recall (TP/(TP+FN)) = {rec:.4f}** — prioritized for clinical safety")

st.markdown("---")

# Predictions preview & download
out_df = df.copy()
out_df["Predicted_Heart_Risk"] = y_pred
if probs is not None:
    out_df["Predicted_Probability"] = probs
st.write("### Predictions (first 10 rows)")
st.dataframe(out_df.head(10))

csv = out_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download predictions CSV", data=csv, file_name="heart_predictions.csv", mime="text/csv")

# Quick model selection helper (advice)
st.markdown("---")
st.write("### Model selection helper")
st.info("For clinical screening choose the model with the highest Recall. Use the threshold slider to tune between recall and specificity.")