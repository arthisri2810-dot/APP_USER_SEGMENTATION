import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "app_user_behavior_dataset.csv")

df = pd.read_csv(DATA_PATH)

# Normalize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Remove duplicates
df.drop_duplicates(inplace=True)

# Data understanding
print(df.info())
print(df.describe())

# Feature selection
features = [
    "sessions_per_week",
    "avg_session_duration_min",
    "daily_active_minutes",
    "feature_clicks_per_session",
    "engagement_score",
    "churn_risk_score"
]

X = df[features].copy()

# Handle missing values
X.fillna(X.mean(), inplace=True)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

# Save processed data
X_scaled_df.to_csv(os.path.join(BASE_DIR, "data", "processed_data.csv"), index=False)

print("✅ Data preprocessing completed successfully")


