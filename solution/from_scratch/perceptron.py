import numpy as np

# trash
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas
from sklearn import datasets


class Sigmoid:
	def __call__(self, x):
		return 1 / (1 + np.exp(-x))

	def gradient(self, x):
		return self.__call__(x) * (1 - self.__call__(x))


class MSE:
	def __call__(self, y_true, y_pred):
		return np.square(y_true - y_pred)

	@staticmethod
	def gradient(y_true, y_pred):
		return -(y_true - y_pred)


class Perceptron:

	def __init__(self, n_iterations=2000, activation_function=Sigmoid, loss=MSE, learning_rate=0.01):
		self.n_iterations = n_iterations
		self.learning_rate = learning_rate
		self.loss = loss()
		self.activation_func = activation_function()
		self.weight = None
		self.biais = None

	def fit(self, X, y):
		n_samples, n_features = np.shape(X)

		self.weight = np.random.uniform(0, 1, n_features)
		self.biais = 0

		for i in range(self.n_iterations):
			linear_output = X.dot(self.weight) + self.biais
			y_pred = self.activation_func(linear_output)

			# Calculate the loss gradient w.r.t the input of the activation function
			error_gradient = self.loss.gradient(y, y_pred) * self.activation_func.gradient(linear_output)

			# Calculate the gradient of the loss with respect to each weight
			grad_wrt_w = X.T.dot(error_gradient)
			grad_wrt_biais = np.sum(error_gradient, axis=0, keepdims=True)

			# Update weights
			self.weight -= self.learning_rate * grad_wrt_w
			self.biais -= self.learning_rate * grad_wrt_biais

	def predict(self, X):
		y_pred = self.activation_func(X.dot(self.weight) + self.biais)
		return y_pred


if __name__ == "__main__":
	clf = Perceptron()

	iris = datasets.load_iris(as_frame=True)
	X = iris.data
	y = iris.target

	# Dictionnaire pour enregistrer les erreurs selon les
	# classifieurs
	erreurs = {}

	# Cette ligne crée une liste contenant toutes les paires possibles
	# entre les 4 mesures.
	# Par exemple : [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
	pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]

	# Reprenons les paires de mesures
	fig, subfigs = plt.subplots(2, 3, tight_layout=True)
	t1 = time.time()
	for (f1, f2), subfig in zip(pairs, subfigs.reshape(-1)):
		f1_name = iris.feature_names[f1]
		f2_name = iris.feature_names[f2]

		# Créez ici un sous-dataset contenant seulement les
		# mesures désignées par f1 et f2
		subdataset = pandas.concat([iris.data[f1_name], iris.data[f2_name]], axis=1)
		X = subdataset.values
		R = iris.target.values

		# Créez ici une grille permettant d'afficher les régions de
		# décision pour chaque classifieur
		h = .015
		x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
		y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
		                     np.arange(y_min, y_max, h))

		# Entraînez le classifieur
		clf.fit(X, R)

		# Utilisez la grille que vous avez créée plus haut
		# pour afficher les régions de décision, de même
		# que les points colorés selon leur vraie classe
		Y = clf.predict(np.c_[xx.ravel(), yy.ravel()])

		cm = plt.cm.RdBu
		colors = ['#FF0000', '#f2faf5', '#0000FF']
		legend_labels = ["Iris Setosa", "Iris Versicolore", "Iris Virginia"]

		Y = Y.reshape(xx.shape)
		subfig.contourf(xx, yy, Y, cmap=cm, alpha=0.8)
		scatter = subfig.scatter(X[:, 0], X[:, 1], c=np.array(colors)[R].tolist(), edgecolor='k', linewidths=1.2)
		subfig.set_xlim(xx.min(), xx.max())
		subfig.set_ylim(yy.min(), yy.max())

		# Identification des axes et des méthodes
		red_patch = matplotlib.lines.Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='red',
		                                    markersize=5, markeredgecolor='k', label=legend_labels[0])
		white_patch = matplotlib.lines.Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='white',
		                                      markersize=5, markeredgecolor='k', label=legend_labels[1])
		blue_patch = matplotlib.lines.Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='blue',
		                                     markersize=5, markeredgecolor='k', label=legend_labels[2])

		subfig.legend(handles=[red_patch, white_patch, blue_patch], fontsize=7)
		subfig.set_xlabel(f1_name)
		subfig.set_ylabel(f2_name)

	plt.show()
	# Affichage des erreurs
	df = pandas.DataFrame(erreurs, index=["KNN"])