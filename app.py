import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix,
    roc_curve, auc
)

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Styling (crimson theme)
# -------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg,#fff0f2 0%, #ffe6ea 50%, #ffdce2 100%);
        font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
        color: #222;
    }
    .hero {
        background: linear-gradient(90deg, rgba(230,57,70,0.10), rgba(220,20,60,0.06));
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 8px 28px rgba(12,24,40,0.06);
        display:flex;
        align-items:center;
        gap:18px;
    }
    .hero h1 { margin:0; font-size:26px; color:#7b1f2f; }
    .subtle { color:#6b3943; margin-top:6px; font-size:13px; }
    .metric-card {
        background: rgba(255,255,255,0.92);
        border-radius: 10px;
        padding: 14px;
        box-shadow: 0 8px 30px rgba(15,20,30,0.04);
        text-align: center;
        border-left: 6px solid rgba(230,57,70,0.95);
        min-height: 110px;                /* ensure equal height */
        display: flex;
        flex-direction: column;
        justify-content: center;         /* vertically center content */
        align-items: center;
    }
    .metric-card h4 { margin: 0 0 6px 0; font-size: 14px; }
    .metric-card h2, .metric-card h3 { margin: 0; font-size: 24px; }
    .metric-small { color:#6b3943; font-size:12px; margin-top:8px; }
    .pill {
        background: rgba(230,57,70,0.12);
        color: #b31f2b;
        padding:6px 10px;
        border-radius:999px;
        font-weight:600;
        font-size:13px;
        display:inline-block;
    }
    .stDownloadButton>button {
        background-color: #e63946;
        color: white;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Better heart SVG (fixed aspect ratio)
# -------------------------
HEART_SVG = """
<svg width="84" height="84" viewBox="0 0 24 24" preserveAspectRatio="xMidYMid meet"
 xmlns="http://www.w3.org/2000/svg" role="img" aria-label="heart">
  <path d="M12 21s-7.5-4.9-9-8.1C1.8 9.2 4 6 7.2 6c1.7 0 3.1.9 4 2.1C12.7 6.9 14.1 6 15.8 6 19 6 21.2 9.2 21 12.9 19.5 16.1 12 21 12 21z"
        fill="rgba(230,57,70,0.14)" stroke="#e63946" stroke-width="0.8"/>
</svg>
"""

# -------------------------
# Model filename map
# -------------------------
MODEL_FILENAME_MAP = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
    "Gradient Boosting": "gradient_boosting.joblib",
}

# -------------------------
# Header/Hero
# -------------------------
with st.container():
    st.markdown(
        f"""
        <div class="hero">
            <div style="flex:0 0 94px;">{HEART_SVG}</div>
            <div>
                <h1>🫀 Heart Disease Risk Predictor</h1>
                <div class="subtle">Clinical screening — RECALL (sensitivity) prioritized to reduce missed cases.</div>
                <div style="height:8px"></div>
                <div class="pill">RECALL-first workflow</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload test CSV (features + Heart_Risk)", type=["csv"])
selected_model_name = st.sidebar.selectbox("Select model", list(MODEL_FILENAME_MAP.keys()))
show_raw = st.sidebar.checkbox("Show raw data table", value=False)
threshold = st.sidebar.slider("Positive probability threshold", 0.0, 1.0, 0.5, 0.01)
st.sidebar.markdown("Tip: lower threshold → higher recall (fewer missed cases).")

# -------------------------
# Helpers
# -------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

def safe_round_int_series(s):
    try:
        return np.where(np.isclose(s, np.round(s)), np.round(s).astype(int), s)
    except Exception:
        return s

# -------------------------
# Early exit if no file
# -------------------------
if uploaded_file is None:
    st.info("Upload a CSV with features + `Heart_Risk` (0/1) to evaluate models.")
    # sample download (if present)
    try:
        with open("test.csv", "rb") as f:
            st.download_button("📥 Download sample test.csv", data=f, file_name="test.csv", mime="text/csv")
    except FileNotFoundError:
        st.warning("Sample file 'test.csv' not found in repo.")

    # show placeholder metric cards so right pane isn't empty
    cols = st.columns(5)
    labels = ["RECALL", "Specificity", "Precision", "F1", "AUC"]
    for c, lab in zip(cols, labels):
        c.markdown(f"""
            <div class="metric-card">
                <h4 style="color:#e63946; margin:0;">{lab if lab=='RECALL' else lab}</h4>
                <h2 style="margin:0;">N/A</h2>
                <div class="metric-small">Awaiting upload</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("### Waiting for data\nUpload a CSV to see confusion matrix, visualizations and downloadable predictions.")
    # do NOT call st.stop() here — keep placeholders visible until a file is uploaded

# -------------------------
# Load CSV
# -------------------------
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

# coerce float->int where appropriate
float_cols = df.select_dtypes(include=["float"]).columns
if len(float_cols) > 0:
    df[float_cols] = df[float_cols].apply(safe_round_int_series)

X_test = df.drop(columns=[target_col])
y_test = df[target_col]

# -------------------------
# Load model
# -------------------------
model_file = MODEL_FILENAME_MAP.get(selected_model_name)
model_path = f"model/{model_file}"
try:
    model = load_model(model_path)
except FileNotFoundError:
    st.error(f"Model file not found: {model_path}. Run `model_training.py` first.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -------------------------
# Predictions (threshold if probabilities)
# -------------------------
try:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        y_pred = (probs >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        probs = None
except Exception as e:
    st.error(f"Prediction error: {e}")
    st.stop()

# -------------------------
# Metrics
# -------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred) if len(np.unique(y_test)) > 1 else 0.0
auc_val = roc_auc_score(y_test, probs) if (probs is not None and len(np.unique(y_test)) > 1) else None

cm = confusion_matrix(y_test, y_pred)
if cm.size == 4:
    tn, fp, fn, tp = cm.ravel()
else:
    tn = fp = fn = tp = 0
specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0

# -------------------------
# Top metric cards (recall emphasized)
# -------------------------
st.markdown("### Performance summary")
cols = st.columns([1.8, 1, 1, 1, 1])
with cols[0]:
    st.markdown(
        f"""
        <div class="metric-card">
            <h4 style="color:#e63946; margin:0;">🔴 RECALL</h4>
            <h2 style="color:#e63946; margin:0;">{rec:.4f}</h2>
            <div class="metric-small">Sensitivity — detected positives / actual positives</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with cols[1]:
    st.markdown(
        f"""
        <div class="metric-card">
            <h4 style="color:#7b1f2f; margin:0;">Specificity</h4>
            <h3 style="margin:0;color:#7b1f2f;">{specificity:.4f}</h3>
            <div class="metric-small">True negative rate</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with cols[2]:
    st.markdown(
        f"""
        <div class="metric-card">
            <h4 style="margin:0;">Precision</h4>
            <h3 style="margin:0;">{prec:.4f}</h3>
            <div class="metric-small">Positive predictive value</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with cols[3]:
    st.markdown(
        f"""
        <div class="metric-card">
            <h4 style="margin:0;">F1</h4>
            <h3 style="margin:0;">{f1:.4f}</h3>
            <div class="metric-small">Harmonic mean of precision & recall</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with cols[4]:
    auc_text = f"{auc_val:.4f}" if auc_val is not None else "N/A"
    st.markdown(
        f"""
        <div class="metric-card">
            <h4 style="margin:0;">AUC</h4>
            <h3 style="margin:0;">{auc_text}</h3>
            <div class="metric-small">ROC AUC (if probabilities)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# -------------------------
# Confusion matrix + clinical notes
# -------------------------
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
    st.markdown(f"- **False Negatives (FN):** {fn} ⚠️ *Missed cases — critical*")
    st.markdown(f"- **False Positives (FP):** {fp}")
    st.markdown(f"- **True Negatives (TN):** {tn}")
    st.markdown(f"**Recall = {rec:.4f}** — prioritized for clinical safety")

st.markdown("---")

# -------------------------
# Plots: prediction distribution, ROC, counts, feature importance
# -------------------------
st.write("### Visualizations")

viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    # Predicted probability histogram (if probabilities available)
    if probs is not None:
        st.write("#### Predicted probability distribution")
        fig, ax = plt.subplots(figsize=(6, 3.2))
        sns.histplot(probs, bins=25, kde=True, color="#b31f2b", ax=ax)
        ax.set_xlabel("Predicted probability (positive)")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    else:
        st.info("Probability distribution unavailable for this model (no predict_proba).")

    # Bar chart of TP/FP/FN/TN counts
    st.write("#### Prediction outcome counts")
    counts = {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette=["#2f855a", "#e53e3e", "#dd6b20", "#2b6cb0"], ax=ax2)
    ax2.set_ylabel("Count")
    st.pyplot(fig2)

with viz_col2:
    # ROC curve (if probabilities available)
    if probs is not None and len(np.unique(y_test)) > 1:
        st.write("#### ROC curve")
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        fig3, ax3 = plt.subplots(figsize=(6, 3.6))
        ax3.plot(fpr, tpr, color="#b31f2b", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        ax3.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
        ax3.set_xlim([0.0, 1.0])
        ax3.set_ylim([0.0, 1.05])
        ax3.set_xlabel("False Positive Rate")
        ax3.set_ylabel("True Positive Rate")
        ax3.legend(loc="lower right")
        st.pyplot(fig3)
    else:
        st.info("ROC curve unavailable (need probabilities and >=2 classes).")

    # Feature importance (if available)
    st.write("#### Feature importances (if available)")
    feature_importances = None
    try:
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            feature_importances = pd.Series(fi, index=X_test.columns)
        elif hasattr(model, "coef_"):
            coef = model.coef_
            # handle multiclass or binary
            if coef.ndim == 1:
                feature_importances = pd.Series(np.abs(coef), index=X_test.columns)
            else:
                feature_importances = pd.Series(np.mean(np.abs(coef), axis=0), index=X_test.columns)
        if feature_importances is not None:
            topk = feature_importances.sort_values(ascending=False).head(8)
            fig4, ax4 = plt.subplots(figsize=(6, 3.6))
            sns.barplot(x=topk.values, y=topk.index, palette="Reds_r", ax=ax4)
            ax4.set_xlabel("Importance")
            st.pyplot(fig4)
        else:
            st.info("Model does not expose feature importances or coefficients.")
    except Exception:
        st.info("Could not compute feature importances for this model.")

st.markdown("---")

# -------------------------
# Predictions preview & download
# -------------------------
out_df = df.copy()
out_df["Predicted_Heart_Risk"] = y_pred
if probs is not None:
    out_df["Predicted_Probability"] = probs

st.write("### Predictions (preview)")
st.dataframe(out_df.head(10))

csv = out_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download predictions", data=csv, file_name="heart_predictions.csv", mime="text/csv")

# -------------------------
# Footer tips
# -------------------------
st.markdown(
    """
    <div style="margin-top:18px;padding:12px;border-radius:10px;background:linear-gradient(90deg,#ffffff,#fff0f2);box-shadow:0 6px 18px rgba(12,24,40,0.03);">
    <strong>Tip:</strong> Adjust the probability threshold to trade between recall and specificity. Lower threshold → higher recall (fewer missed cases).
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
<style>
section[data-testid="stSidebar"], div[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #ff6b6b 0%, #ffccd5 100%) !important;
  border-radius: 12px;
  padding: 10px;
  box-shadow: 0 8px 20px rgba(255,107,107,0.06);
}
</style>
""", unsafe_allow_html=True)