import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
from PIL import Image

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

# Load model
model = joblib.load("Models/LogisticRegression_churn_model.pkl")

# Title
st.title("🧠 Customer Churn Prediction Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload CSV for Prediction", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Auto-detect target col and remove it
    target_col = [col for col in df.columns if "churn" in col.lower()]
    if target_col:
        df.drop(columns=target_col, inplace=True)

    # Predict probabilities
    probs = model.predict_proba(df)[:, 1]
    df["Churn_Probability"] = probs
    df_sorted = df.sort_values("Churn_Probability", ascending=False)

    st.subheader("🔢 Top 10 High-Risk Customers")
    st.dataframe(df_sorted.head(10))

    # Plot 1: Churn probability histogram
    st.subheader("📊 Churn Probability Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(data=df, x="Churn_Probability", bins=20, kde=True, ax=ax1, color="tomato")
    st.pyplot(fig1)

    # Plot 2: Churn vs Retain Pie
    st.subheader("🧩 Churn vs Retain (Assuming 0.5 Threshold)")
    df["Predicted_Churn"] = (df["Churn_Probability"] >= 0.5).astype(int)
    pie_data = df["Predicted_Churn"].value_counts().rename({0: "Retain", 1: "Churn"})
    fig2, ax2 = plt.subplots()
    ax2.pie(pie_data, labels=list(pie_data.index.astype(str)), autopct='%1.1f%%', startangle=90, colors=["skyblue", "salmon"])
    ax2.axis('equal')
    st.pyplot(fig2)

    # Downloadable predictions
    st.subheader("⬇️ Download Predictions")
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "churn_predictions.csv", "text/csv")

else:
    st.info("Upload a preprocessed CSV file to begin.")

#  SHAP Plots 
    st.subheader("🔍 Model Explainability using SHAP")

    shap_dir = "reports/shap"
    shap_bar = os.path.join(shap_dir, "shap_summary_bar.png")
    shap_beeswarm = os.path.join(shap_dir, "shap_summary_beeswarm.png")

    if os.path.exists(shap_bar) and os.path.exists(shap_beeswarm):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📊 Feature Importance (Bar)")
            st.image(Image.open(shap_bar), use_container_width=True)

        with col2:
            st.markdown("#### 🐝 Beeswarm Plot")
            st.image(Image.open(shap_beeswarm), use_container_width=True)
    else:
        st.warning("SHAP plots not found. Please run SHAP analysis first.")

st.subheader("📊 Model Comparison (AUC, Accuracy, F1)")

if os.path.exists("reports/metrics.csv"):
    metrics_df = pd.read_csv("reports/metrics.csv")
    st.dataframe(metrics_df)
    best_model_name = metrics_df.iloc[0]["Model"]
    st.success(f"🏆 Best Performing Model: {best_model_name}")
else:
    st.warning("No model comparison report found.")



