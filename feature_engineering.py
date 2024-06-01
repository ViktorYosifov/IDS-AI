from sklearn.decomposition import PCA

def reduce_dimensionality(X, n_components=30):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca
