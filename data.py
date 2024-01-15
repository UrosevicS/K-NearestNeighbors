import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def create_data(k, n_features=2, n_samples=1000):
    n_samples_per_k = n_samples // k

    # we do it like these in case n_samples is not divisible by k
    X = np.zeros((n_samples_per_k * k, n_features))
    y = np.zeros((n_samples_per_k * k,))

    for label in range(k):
        features = np.random.uniform(size=(n_samples_per_k, n_features))
        for i in range(n_features):
            features[:, i] *= np.random.randint(30,50)
            features[:, i] += (np.random.randint(-5, 5) * (label + 1)*5) + 8*np.random.uniform(size=(n_samples_per_k,))
        X[n_samples_per_k * label:n_samples_per_k * (label + 1)] = features
        y[n_samples_per_k * label:n_samples_per_k * (label + 1)] = label

    X, y = shuffle(X, y, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    return X_train, X_test, y_train, y_test


def visualize_data(x, y):
    colors = ['r', 'b', 'y']
    for i in range(len(x)):
        plt.scatter(*x[i], c=colors[int(y[i])], marker='o', label='Dots')
    plt.show()


if __name__ == '__main__':
    X, y, _, _ = create_data(3)
    visualize_data(X, y)
