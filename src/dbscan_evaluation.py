import pandas as pd

# Load DBSCAN clustered data
df = pd.read_csv("outputs/dbscan_clustered_users.csv")

print("DBSCAN Cluster Counts:")
print(df['dbscan_cluster'].value_counts())

# Select numeric columns only
numeric_df = df.select_dtypes(include='number')

print("\nDBSCAN Cluster Profiles (Numeric Features Only):")
print(numeric_df.groupby(df['dbscan_cluster']).mean())
