import numpy as np


class KNN:
	def __init__(self, k=15):
		self.k = k
		self.X = None
		self.y = None

	def fit(self, X, y):
		self.X = X
		self.y = y

	def predict(self, X):
		out = []
		for x in X:
			dist = np.sum((self.X - x)**2, axis=-1)
			sorted_idx = np.argsort(dist)
			k_indexes = sorted_idx[:self.k]
			y_nn = self.y[k_indexes]
			y_nn, y_nn_idx = np.unique(y_nn, return_counts=True)
			out.append(y_nn[np.argmax(y_nn_idx)])
		return np.asarray(out)



