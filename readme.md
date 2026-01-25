App User Behavior Segmentation
Unsupervised Learning using BIRCH, DBSCAN, and PCA
Project Overview

This project focuses on segmenting app users based on their behavioral patterns using unsupervised machine learning techniques.
Since no labeled target variable is available, clustering algorithms are used to discover hidden user groups.

The project applies BIRCH as the final clustering model, DBSCAN as a comparison algorithm, and PCA for visualization and validation.

Objectives

Segment users without labeled data

Apply unsupervised clustering (non–K-Means)

Compare different clustering algorithms

Interpret clusters for business insights

Handle a large dataset (50,000 users) efficiently

Dataset Description

Total records: 50,000 users

Domain: App user behavior analytics

Key numerical features used:

sessions_per_week

avg_session_duration_min

daily_active_minutes

feature_clicks_per_session

notifications_opened_per_week

in_app_search_count

pages_viewed_per_session

ads_clicked_last_30_days

content_downloads

social_shares

days_since_last_login

crash_events_last_30_days

support_tickets_raised

engagement_score

churn_risk_score

account_age_days

Only numeric behavioral features were used for clustering and evaluation.

Algorithms Used
BIRCH Clustering (Final Model)

Hierarchical and scalable clustering algorithm

Suitable for large datasets

Produces stable and interpretable clusters

Used as the final segmentation method

DBSCAN (Comparison Model)

Density-based clustering algorithm

Does not require predefined number of clusters

Effective for outlier detection

Used only for comparison, not final segmentation

3️PCA (Principal Component Analysis)

Used only for visualization

Reduces high-dimensional data to 2D

Helps validate clustering behavior

Project Workflow

Load and inspect dataset

Select relevant numeric behavioral features

Handle missing values

Scale features using StandardScaler

Apply BIRCH clustering

Apply DBSCAN for comparison

Evaluate clusters using mean profiling

Visualize clusters using PCA

Results & Interpretation
BIRCH Clustering Results

BIRCH successfully segmented users into four distinct clusters:

Cluster	User Segment	Description
0	Moderate Engagement Users	Balanced and consistent usage
1	High Engagement Users	Frequent usage, high engagement
2	Low Engagement / At-Risk Users	Low activity, higher churn risk
3	Occasional Users	Irregular or seasonal usage

Cluster meanings were derived from cluster-wise mean profiling, not from labels.

DBSCAN Results (Comparison)

DBSCAN identified:

One dense cluster

Noise points (-1)

Noise users represent irregular or anomalous behavior

DBSCAN was effective for outlier detection but not suitable for multi-class user segmentation.

PCA Visualization

BIRCH PCA: Shows structured clusters (some overlap expected due to dimensionality reduction)

DBSCAN PCA: Shows one dense cluster and scattered noise points

PCA was used only for visualization, not for clustering.

How to Run the Project
Activate virtual environment
.\.venv\Scripts\Activate.ps1

Run BIRCH clustering (final model)
python src/final_clustering.py

Evaluate BIRCH clusters
python src/evaluation.py

Run DBSCAN (comparison)
python src/dbscan.py
python src/dbscan_evaluation.py

Generate PCA visualizations
python src/pca_visualization_both.py

Business Use Cases

User segmentation for marketing campaigns

Early churn detection

Personalized recommendations

Engagement optimization

Product and feature analysis

Conclusion

This project demonstrates effective use of unsupervised machine learning to segment large-scale app user data.
While DBSCAN helped identify outliers, BIRCH proved more suitable for scalable and interpretable user segmentation.
PCA visualization supported the validation of clustering behavior.
 BIRCH was selected as the final model.