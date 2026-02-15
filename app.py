import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
 
 
@st.cache_resource
def load_model(path):
    if not path:
        return None
    return joblib.load(path)
 
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
 
st.title("🫀 Heart Disease Prediction System")
st.markdown("Upload a test dataset to evaluate different machine learning models.")
 
# --- Sidebar ---
st.sidebar.header("Configuration")
 
# 1. Dataset Upload
uploaded_file = st.sidebar.file_uploader("Upload CSV (Test Data)", type=["csv"])
 
# 2. Model Selection
model_options = [
    "Logistic_Regression", 
    "Decision_Tree", 
    "kNN", 
    "Naive_Bayes", 
    "Random_Forest", 
    "XGBoost"
]
selected_model_name = st.sidebar.selectbox("Select Model", model_options)
 
# --- Main Content ---
 
if uploaded_file is not None:
    # Load Data
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### 📊 Uploaded Dataset Preview")
        st.dataframe(df.head())
        # Check for Target Column
        target_col = 'Heart_Risk'  # Update this if your target column has a different name
        if target_col not in df.columns:
            st.error(f"❌ Dataset must contain the target column '{target_col}' for evaluation.")
        else:
            float_cols = df.select_dtypes(include=float).columns
            df[float_cols] = df[float_cols].astype(int)
 
            # Prepare X and y
            X_test = df.drop(target_col, axis=1)
            y_test = df[target_col]
            # Load Model
            model_path = f"model/{selected_model_name.lower()}.joblib"
            try:
                pipeline = load_model(model_path)
                # Predict
                y_pred = pipeline.predict(X_test)
                # Get Probabilities (if supported) for AUC
                if hasattr(pipeline, "predict_proba"):
                    y_prob = pipeline.predict_proba(X_test)[:, 1]
                else:
                    y_prob = None
                # --- Metrics Calculation ---
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                mcc = matthews_corrcoef(y_test, y_pred)
                auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.5
                # --- Display Metrics ---
                st.write(f"### 🚀 Performance of {selected_model_name}")
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                col1.metric("Accuracy", f"{acc:.4f}")
                col2.metric("AUC Score", f"{auc:.4f}")
                col3.metric("Precision", f"{prec:.4f}")
                col4.metric("Recall", f"{rec:.4f}")
                col5.metric("F1 Score", f"{f1:.4f}")
                col6.metric("MCC", f"{mcc:.4f}")
                # --- Confusion Matrix ---
                st.write("### 🧩 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title(f'Confusion Matrix - {selected_model_name}')
                st.pyplot(fig)
                # --- Classification Report Download ---
                # Optional: Provide a download of predictions
                df['Predicted_Status'] = y_pred
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv',
                )
            except FileNotFoundError:
                st.error(f"⚠️ Model file '{model_path}' not found. Please run 'model_training.py' first.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
 
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
 
else:
    st.info("👋 Please upload a CSV file in the sidebar to begin analysis.")
    st.text("Download a sample from here")
    file_path = "test.csv"
    with open(file_path, "rb") as file:
        st.download_button(
            label="📥 Download Sample CSV",
            data=file,
            file_name="test.csv",
            mime="text/csv"
        )
 
    st.markdown("""
    **Expected CSV Format:**
    - Must contain feature columns: Chest_Pain', 'Shortness_of_Breath', 'Fatigue', 'Palpitations',
       'Dizziness', 'Swelling', 'Pain_Arms_Jaw_Back', 'Cold_Sweats_Nausea',
       'High_BP', 'High_Cholesterol', 'Diabetes', 'Smoking', 'Obesity',
       'Sedentary_Lifestyle', 'Family_History', 'Chronic_Stress', 'Gender',
       'Age'.
    - Must contain target column: `Heart_Risk` (for evaluation metrics).
    """)