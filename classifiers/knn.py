import pickle

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


KNN_FILENAME = './resources/knn-classifier.pkl'


def train_knn(train_features: np.ndarray, train_label: np.ndarray) -> KNeighborsClassifier:
    knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    knn.fit(train_features, train_label)
    return knn


def load_model():
    return pickle.load(open(KNN_FILENAME, 'rb'))

