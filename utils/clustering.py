from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def find_optimal_clusters(data):
    inertia = []
    K = range(1, 10)

    for k in K:
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(data)
        inertia.append(model.inertia_)

    plt.plot(K, inertia, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.show()

def apply_kmeans(data, k=4):
    model = KMeans(n_clusters=k, random_state=42)
    clusters = model.fit_predict(data)
    return clusters, model