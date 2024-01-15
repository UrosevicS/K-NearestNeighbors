import numpy as np


class KNN:
    def __init__(self, k):
        self.k = k

    def train(self, X, y):
        # X = [[f1, f2, ... fn],
        #      [f1, f2, ... fn],
        #      [f1, f2, ... fn]...]
        # y = [1,
        #      1,
        #      2,
        #      0,
        #      1,
        #      0,...]
        self.X_train = X
        self.labels = y

    def predict(self, X):
        distances = self.get_distances(X)
        prediction = self.predict_label(distances)
        return prediction

    def get_distances_in_slower_way(self, X):
        # euclidian distance
        # F = train features, X = test feature
        # sqrt((X1 - F1)**2 + (X2 - F2)**2 ...)
        distances = np.zeros((X.shape[0], self.X_train.shape[0]))
        for i in range(X.shape[0]):
            distances[i,:] = np.sqrt(np.sum((self.X_train - X[i])**2, axis=1))
        # distances = np.sqrt(np.sum((self.features - X) ** 2, axis=1))
        return distances

    def get_distances(self, X):
        distances = np.sum(X**2, axis=1, keepdims=True) - 2*np.dot(X, self.X_train.T) + np.sum(self.X_train**2, axis=1, keepdims=True).T
        return np.sqrt(distances)

    def predict_label(self, distances):
        num_test = distances.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            y_indices = np.argsort(distances[i, :])
            k_closest_classes = self.labels[y_indices[: self.k]].astype(int)
            y_pred[i] = np.argmax(np.bincount(k_closest_classes))

        return y_pred

if __name__ == '__main__':
    from data import create_data, visualize_data

    k = 3
    X_train, X_test, y_train, y_test  = create_data(k=k)
    model = KNN(k)
    model.train(X_train, y_train)
    # visualize_data(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy: {sum(y_pred == y_test) / y_test.shape[0]}")