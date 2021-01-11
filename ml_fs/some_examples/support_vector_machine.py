from __future__ import division, print_function
import numpy as np
from sklearn import datasets

# Import helper functions
from ml_fs.common_utils import train_test_split, normalize, accuracy_score, Plot
from ml_fs.common_utils.kernels import *
from ml_fs.supervised_learning_models import SupportVectorMachine

def main():
    data = datasets.load_iris()
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    y[y == 1] = -1
    y[y == 2] = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    clf = SupportVectorMachine(kernel=polynomial_kernel, power=4, coef=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

    # Reduce dimension to two using PCA and plot the results
    Plot().plot_in_2d(X_test, y_pred, title="Support Vector Machine", accuracy=accuracy)


if __name__ == "__main__":
    main()
