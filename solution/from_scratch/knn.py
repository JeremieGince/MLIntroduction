import numpy as np
from typing import Optional


class KNN:
	def __init__(self, k=15):
		self.k = k
		self.X = None
		self.y = None

	def fit(self, X: np.ndarray, y: np.ndarray):
		self.X = X
		self.y = y

	def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
		out = []
		for x in X:
			dist = np.sum((self.X - x)**2, axis=-1)
			sorted_idx = np.argsort(dist)
			k_indexes = sorted_idx[:self.k]
			y_nn = self.y[k_indexes]
			y_nn, y_nn_idx = np.unique(y_nn, return_counts=True)
			out.append(y_nn[np.argmax(y_nn_idx)])
		out = np.asarray(out)
		if y is not None:
			acc = np.isclose(np.abs(y - out), 0).astype(dtype=int).sum() / y.shape[0]
			return out, acc
		return out



