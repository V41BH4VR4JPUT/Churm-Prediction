import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
import os

def train_and_evaluate_models(data_path, target_col="churned"):
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    metrics = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        # Save model
        joblib.dump(model, f"Models/{name}_model.pkl")

        metrics.append({
            "Model": name,
            "Accuracy": acc,
            "F1 Score": f1,
            "AUC-ROC": auc
        })

    # Ensemble
    ensemble = VotingClassifier(estimators=[
        ('lr', models["LogisticRegression"]),
        ('rf', models["RandomForest"]),
        ('xgb', models["XGBoost"])
    ], voting='soft')

    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    joblib.dump(ensemble, "Models/Ensemble_model.pkl")

    metrics.append({
        "Model": "Ensemble (Voting)",
        "Accuracy": acc,
        "F1 Score": f1,
        "AUC-ROC": auc
    })

    metrics_df = pd.DataFrame(metrics).sort_values("AUC-ROC", ascending=False)
    os.makedirs("reports", exist_ok=True)
    metrics_df.to_csv("reports/metrics.csv", index=False)
    print(metrics_df)

    return metrics_df

train_and_evaluate_models(
    data_path="Data/processed_train.csv",
)