import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

def explain_model(model_path, data_path, output_dir='reports/shap/'):
    # Load data and model
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)

    # Auto-detect target
    target_col = [col for col in df.columns if 'churn' in col.lower()][0]
    X = df.drop(columns=[target_col])

    # Create output dir
    os.makedirs(output_dir, exist_ok=True)

    # SHAP init
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Summary Plot (Bar)
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.title("SHAP Feature Importance (Bar)")
    plt.savefig(f"{output_dir}shap_summary_bar.png")
    print(f" SHAP summary bar plot saved: {output_dir}shap_summary_bar.png")

    # Summary Plot (Beeswarm)
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.title("SHAP Feature Importance (Beeswarm)")
    plt.savefig(f"{output_dir}shap_summary_beeswarm.png")
    print(f" SHAP beeswarm plot saved: {output_dir}shap_summary_beeswarm.png")

    # Force plot for first prediction (optional)
    # shap.initjs()
    # shap.plots.force(shap_values[0])  # for Jupyter only

    return shap_values

explain_model(
    model_path="Models/LogisticRegression_churn_model.pkl", 
    data_path="Data/processed_train.csv"
)