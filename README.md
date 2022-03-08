# MLIntroduction

---------------------------------------------------------------------------
Exercice d'introduction à l'apprentissage machine.

## Instructions:
 1. Dans un premier temps, vous devez adapter le code de "knn_iris.ipynb" pour entraîner un perceptron de sklearn sur le 
    dataset de Iris. On vous suggère de comparer les performances des deux algorithmes. Votre implémentation sera fait 
    dans le fichier "exercice/sklearn/perceptron_iris.py".
 2. Ensuite, vous allez devoir implémenter un K-NN avec seulement le package numpy à votre disposition dans le fichier
    "exercice/from_scratch/knn.py". Faite vous un objet KNN ayant les méthodes suivantes qui sont basé sur le template 
    de sklearn:
    1. ```fit(X: np.ndarray, y: np.ndarray) -> None```
    2. ```predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray```
 3. Refaite le même exercice, mais avec le perceptron dans le fichier "exercice/from_scratch/perceptron.py".
 4. Finalement, vous pouvez comparer les résultats des algorithmes en les entraînant sur le dataset 
    [digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) de sklearn.
    la fonction suivante vous sera utile pour downloader le dataset 
    ```X, y = datasets.load_digits(return_X_y=True)```.
    1. Vous aurez à calculer la [matrice de confusion](https://en.wikipedia.org/wiki/Confusion_matrix) de la 
       classification des classifications. Afficher les sous forme de heatmap ou d'image afin de pouvoir les visualiser.
    2. De plus, calculer les [métriques](https://en.wikipedia.org/wiki/Precision_and_recall) suivantes pour chaque 
       classifieur:
       1. Accuracy
       2. Recall
       3. F1Score



## Setup

- Cloner le répertoire présent.
- Créer votre environnement virtuel pour ces exercices.
- Installer les dépendances avec 
  - ```pip install -r requirements.txt```


## Références
- Pour plus d'information sur comment utiliser git:
    - [TutorielPython-Manuel/git](https://github.com/JeremieGince/TutorielPython-Manuel/tree/master/Cycle-de-developpement-avec-git)
- Pour plus d'information sur comment créer un environnement virtuel:
    - [TutorielPython-Manuel/Environments](https://github.com/JeremieGince/TutorielPython-Manuel/tree/master/Environments)
- Si vous désirez avoir des ressources au niveau de l'affichage avec python:
  - [Atelier de visualisation du ProgFest](https://github.com/rem657/AtelierVisualisation)


## Solution
La solution est fournie dans le dossier './solution'.



---------------------------------------------------------------------------

<p align="center"> <img src="https://github.com/JeremieGince/MLIntroduction/blob/main/images/progfest_logo.png?raw=true"> </p>