import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_csv("data/app_user_behavior_dataset.csv")

print("Dataset loaded successfully")

# ===============================
# 2. Feature Selection (DATASET-CORRECT)
# ===============================
features = [
    'sessions_per_week',
    'avg_session_duration_min',
    'feature_clicks_per_session',
    'engagement_score',
    'churn_risk_score'
]

X = df[features].fillna(df[features].mean())

print("Feature selection completed")

# ===============================
# 3. Scaling
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Feature scaling completed")

# ===============================
# 4. DBSCAN Clustering
# ===============================
dbscan = DBSCAN(
    eps=1.3,
    min_samples=10,
    n_jobs=-1
)

df['dbscan_cluster'] = dbscan.fit_predict(X_scaled)

print("DBSCAN clustering completed")

# ===============================
# 5. PCA Visualization
# ===============================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df['dbscan_cluster'],
    cmap='tab10',
    s=10
)
plt.title("DBSCAN User Segmentation (PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.tight_layout()
plt.show()

# ===============================
# 6. Save Output
# ===============================
df.to_csv("outputs/dbscan_clustered_users.csv", index=False)

print("DBSCAN results saved to outputs/dbscan_clustered_users.csv")

