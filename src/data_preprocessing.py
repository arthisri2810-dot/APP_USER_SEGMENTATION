import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "app_user_behavior_dataset.csv")

df = pd.read_csv(DATA_PATH)

# Normalize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

features = [
    "sessions_per_week",
    "avg_session_duration_min",
    "daily_active_minutes",
    "feature_clicks_per_session",
    "engagement_score",
    "churn_risk_score"
]

X = df[features].copy()
X.fillna(X.mean(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✅ Data preprocessing completed successfully")


