import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

print("STEP 1: Script started", flush=True)

# Ensure output folder exists
os.makedirs("outputs", exist_ok=True)

# Load Dataset
df = pd.read_csv("data/app_user_behavior_dataset.csv")
print("STEP 2: Dataset loaded", df.shape, flush=True)

# Feature Selection
features = [
    'sessions_per_week',
    'avg_session_duration_min',
    'daily_active_minutes',
    'feature_clicks_per_session',
    'engagement_score',
    'churn_risk_score',
    'days_since_last_login'
]

X = df[features].fillna(df[features].mean())
print("STEP 3: Feature selection completed", flush=True)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("STEP 4: Scaling completed", flush=True)

# Elbow Method
inertia = []
K = range(1, 10)

for k in K:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

plt.plot(K, inertia, marker='o')
plt.title("Elbow Method")
plt.show()

# KMeans Clustering
print("STEP 5: Starting KMeans clustering", flush=True)

kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_scaled)

print("STEP 6: KMeans clustering completed", flush=True)

# Add clusters
df['cluster'] = labels

# Cluster Naming
cluster_names = {
    0: "High Engagement",
    1: "Moderate Users",
    2: "Low Engagement",
    3: "Occasional Users"
}
df['segment'] = df['cluster'].map(cluster_names)

# Cluster Profiling
print("\nCluster Profiles:")
print(df.groupby('cluster')[features].mean())

# PCA Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
plt.title("PCA Clusters")
plt.show()

# Evaluation
score = silhouette_score(X_scaled, labels)
print("Silhouette Score:", score)

# Save Output
df.to_csv("outputs/final_clustered_users.csv", index=False)

print("STEP 7: Output saved", flush=True)
print("DONE ✅", flush=True)
print("\n📊 BUSINESS ACTION SUMMARY")

print("""
High Engagement → Loyalty & premium offers
Moderate → Personalization strategies
At-Risk → Retention campaigns
Occasional → Re-engagement strategies
""")