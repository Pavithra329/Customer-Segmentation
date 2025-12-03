from sklearn.decomposition import PCA

def apply_pca(data, n_components=3):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(data)
    return reduced, pca
