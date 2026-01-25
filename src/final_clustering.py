import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch

print("STEP 1: Script started", flush=True)

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("data/app_user_behavior_dataset.csv")
print("STEP 2: Dataset loaded", df.shape, flush=True)

# ===============================
# Feature Selection
# ===============================
features = [
    'sessions_per_week',
    'avg_session_duration_min',
    'daily_active_minutes',
    'feature_clicks_per_session',
    'notifications_opened_per_week',
    'in_app_search_count',
    'pages_viewed_per_session',
    'ads_clicked_last_30_days',
    'content_downloads',
    'social_shares',
    'days_since_last_login',
    'crash_events_last_30_days',
    'support_tickets_raised',
    'engagement_score',
    'churn_risk_score',
    'account_age_days'
]

X = df[features].fillna(df[features].mean())
print("STEP 3: Feature selection completed", flush=True)

# ===============================
# Scaling
# ===============================
X_scaled = StandardScaler().fit_transform(X)
print("STEP 4: Scaling completed", flush=True)

# ===============================
# BIRCH Clustering (SAFE)
# ===============================
print("STEP 5: Starting BIRCH clustering", flush=True)

birch = Birch(
    threshold=0.8,      # high threshold = fast
    branching_factor=50,
    n_clusters=4
)

labels = birch.fit_predict(X_scaled)

print("STEP 6: BIRCH clustering completed", flush=True)

# ===============================
# Save Output
# ===============================
df['cluster'] = labels
df.to_csv("outputs/final_clustered_users.csv", index=False)

print("STEP 7: Output saved", flush=True)
print("DONE ✅", flush=True)