import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from solution.from_scratch.knn import KNN
from solution.from_scratch.perceptron import Perceptron

if __name__ == '__main__':
	clf_dict = dict(
		knn=KNN(),
		perceptron=Perceptron(),
	)
	
	X, y = datasets.load_digits(return_X_y=True)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	
	df = pd.DataFrame(columns=list(clf_dict.keys()), index=['acc', 'recall', 'f1'])
	
	figure, axes = plt.subplots(1, 2, figsize=(12, 6))
	axes = np.ravel(axes)
	for i, (clf_name, clf) in enumerate(clf_dict.items()):
		# Entraînez le classifieur
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		cm = confusion_matrix(y_test, y_pred)
		df.loc['acc', clf_name] = accuracy_score(y_test, y_pred)
		df.loc['recall', clf_name] = recall_score(y_test, y_pred, average='macro')
		df.loc['f1', clf_name] = f1_score(y_test, y_pred, average='macro')
		axes[i].imshow(cm)
		axes[i].set_title(clf_name)
		axes[i].set_xlabel("Classes [-]")
		axes[i].set_ylabel("Classes [-]")
	
	print(df)
	plt.show()
