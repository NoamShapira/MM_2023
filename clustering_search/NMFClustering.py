import pandas as pd
from sklearn.base import ClusterMixin


class NMFClustering(ClusterMixin):
    def __init__(self, nmf_model):
        self.model = nmf_model
        self.W = None
        self.H = None

    def _fit(self, X):
        self.W = self.model.fit_transform(X)
        self.H = self.model.components_

    def _predict_clusters(self):
        return self.W.argmax(axis=1)

    def fit_predict(self, X, y=None, **kwargs):
        self._fit(X)

        if isinstance(X, pd.DataFrame):
            clust = pd.Series(self._predict_clusters(), index=X.index)
        else:
            clust = self._predict_clusters()
        return clust