from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from data import create_data, visualize_data

k = 3
X_train, X_test, y_train, y_test = create_data(k=k)
knn_classifier = KNeighborsClassifier(n_neighbors=k)

knn_classifier.fit(X_train, y_train)

y_pred = knn_classifier.predict(X_test)

accuracy = sum(y_pred == y_test)/y_test.shape[0]

print(f"Accuracy: {accuracy}")