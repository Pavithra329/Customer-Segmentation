from sklearn.cluster import KMeans
from minisom import MiniSom
import numpy as np

def kmeans_clustering(data, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels, kmeans

def som_clustering(data, x=10, y=10, iterations=1000):
    som = MiniSom(x, y, data.shape[1], sigma=1.0, learning_rate=0.5)
    som.random_weights_init(data)
    som.train_random(data, iterations)
    labels = np.array([som.winner(d) for d in data])
    return labels, som
