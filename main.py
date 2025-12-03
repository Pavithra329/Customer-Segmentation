from src.preprocess import load_data, feature_engineering, scale_features
from src.dimensionality import apply_pca
from src.clustering import kmeans_clustering, som_clustering
from src.visualization import plot_3d_clusters

if __name__ == "__main__":
    df = load_data('data/transactions.csv')
    df = feature_engineering(df)
    scaled_data = scale_features(df)
    
    pca_data, _ = apply_pca(scaled_data, n_components=3)

    k_labels, _ = kmeans_clustering(pca_data, n_clusters=4)
    som_labels, _ = som_clustering(pca_data, x=10, y=10)

    plot_3d_clusters(pca_data, k_labels, title="K-Means Clusters")
    plot_3d_clusters(pca_data, som_labels[:,0]*10+som_labels[:,1], title="SOM Clusters")
