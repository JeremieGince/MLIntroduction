import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import datasets, neighbors
from solution.from_scratch.knn import KNN


if __name__ == '__main__':
	iris = datasets.load_iris()
	X = iris.data
	y = iris.target

	clf = KNN()
	clf.fit(X, y)
	clf.predict(X[:5])



