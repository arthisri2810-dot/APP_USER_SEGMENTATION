import pandas as pd

# Load FINAL clustered output
df = pd.read_csv("outputs/final_clustered_users.csv")

# Cluster profiling
profile = df.groupby('cluster').mean(numeric_only=True)

print("Cluster Profiles:\n")
print(profile)
