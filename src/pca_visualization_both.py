import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ===============================
# Common feature list (dataset-correct)
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

def run_pca(df, cluster_col, title, output_file, noise_label=None):
    X = df[features]
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))

    # If DBSCAN, separate noise
    if noise_label is not None:
        noise = df[cluster_col] == noise_label
        core = df[cluster_col] != noise_label

        plt.scatter(
            X_pca[noise, 0],
            X_pca[noise, 1],
            c='gray',
            s=8,
            label='Noise (-1)'
        )

        plt.scatter(
            X_pca[core, 0],
            X_pca[core, 1],
            c=df.loc[core, cluster_col],
            cmap='tab10',
            s=10,
            label='Clusters'
        )
        plt.legend()
    else:
        # BIRCH
        plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=df[cluster_col],
            cmap='viridis',
            s=10
        )

    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig(f"outputs/{output_file}")
    plt.show()


# ===============================
# 1️⃣ PCA FOR BIRCH
# ===============================
birch_df = pd.read_csv("outputs/final_clustered_users.csv")

run_pca(
    birch_df,
    cluster_col='cluster',
    title="PCA Visualization – BIRCH Clustering",
    output_file="pca_birch_clusters.png"
)

print("BIRCH PCA completed")


# ===============================
# 2️⃣ PCA FOR DBSCAN
# ===============================
dbscan_df = pd.read_csv("outputs/dbscan_clustered_users.csv")

run_pca(
    dbscan_df,
    cluster_col='dbscan_cluster',
    title="PCA Visualization – DBSCAN Clustering",
    output_file="pca_dbscan_clusters.png",
    noise_label=-1
)

print("DBSCAN PCA completed")
