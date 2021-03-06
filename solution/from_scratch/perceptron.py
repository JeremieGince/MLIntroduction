import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def normalize(X, axis=-1, order=2):
	l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
	l2[l2 == 0] = 1
	return X / np.expand_dims(l2, axis)


def to_categorical(x, n_col=None):
	if not n_col:
		n_col = np.amax(x) + 1

	one_hot = np.zeros((x.shape[0], n_col))
	one_hot[np.arange(x.shape[0]), x] = 1
	return one_hot


class Sigmoid:
	def __call__(self, x):
		return 1 / (1 + np.exp(-x))

	def gradient(self, x):
		return self(x)*(1 - self(x))


class CrossEntropy:
	def __init__(self):
		pass

	def loss(self, y, p):
		p = np.clip(p, 1e-15, 1 - 1e-15)
		return -y * np.log(p) - (1 - y) * np.log(1 - p)

	def acc(self, y, p):
		return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

	def gradient(self, y, p):
		p = np.clip(p, 1e-15, 1 - 1e-15)
		return -(y / p) + (1 - y) / (1 - p)


class Perceptron:

	def __init__(self, n_iterations=10000, activation_function=Sigmoid, loss=CrossEntropy, learning_rate=0.01):
		self.n_iterations = n_iterations
		self.learning_rate = learning_rate
		self.loss = loss()
		self.activation_func = activation_function()
		self.weight = None
		self.biais = None

	def fit(self, X, y):
		# One-hot encoding of nominal y-values
		y = to_categorical(y)

		n_samples, n_features = np.shape(X)

		_, n_outputs = np.shape(y)

		limit = 1 / np.sqrt(n_features)
		self.weight = np.random.uniform(-limit, limit, (n_features, n_outputs))
		self.biais = np.zeros((1, n_outputs))

		for i in range(self.n_iterations):
			linear_output = X.dot(self.weight) + self.biais
			y_pred = self.activation_func(linear_output)

			# Calculate the loss gradient w.r.t the input of the activation function
			error_gradient = self.loss.gradient(y, y_pred) * self.activation_func.gradient(y_pred)

			# Calculate the gradient of the loss with respect to each weight
			grad_wrt_w = X.T.dot(error_gradient)
			grad_wrt_biais = np.sum(error_gradient, axis=0, keepdims=True)

			# Update weights
			self.weight -= self.learning_rate * grad_wrt_w
			self.biais -= self.learning_rate * grad_wrt_biais

	def predict(self, X):
		y_pred = self.activation_func(X.dot(self.weight) + self.biais)
		y_pred = np.argmax(y_pred, axis=-1)
		return y_pred
