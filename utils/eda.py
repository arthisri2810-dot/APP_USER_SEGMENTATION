import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution(df, col):
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

def correlation_map(df):
    sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("data/app_user_behavior_dataset.csv")

    plot_distribution(df, 'engagement_score')
    correlation_map(df)
