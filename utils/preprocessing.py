import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(file_path , save_path="Data/processed_train.csv"):
    df = pd.read_csv(file_path)
    df.drop_duplicates(inplace=True)
    df.columns = df.columns.str.strip()
    df.ffill(inplace=True)

    target_col = 'churned'
    y = df[target_col]
    X = df.drop(columns=[target_col])

    Categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    Numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    label_encoders = {}
    for col in Categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    X[Numerical_cols] = scaler.fit_transform(X[Numerical_cols])

    joblib.dump(label_encoders, "Models/label_encoders.pkl")
    joblib.dump(scaler, "Models/scaler.pkl")

    processed_df = X.copy()
    processed_df[target_col] = y

    processed_df.to_csv(save_path, index=False)
    print(f"Data preprocessing complete. Processed data saved to {save_path}")

    return processed_df , X , y    

df , X, y = preprocess_data("Data/Data.csv")