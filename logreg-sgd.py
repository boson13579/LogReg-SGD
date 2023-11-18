from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import sys

import math
import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing
from random import randint as rd
from functools import cmp_to_key


def load_train_test_data(train_ratio=0.5):
    data = pandas.read_csv(
        "./HTRU_2.csv", header=None, names=["x%i" % (i) for i in range(8)] + ["y"]
    )
    X = numpy.asarray(data[["x%i" % (i) for i in range(8)]])
    # I can't understand what the following line does, so I just commented it out.
    # X = numpy.hstack((numpy.ones((X.shape[0],1)), X))
    y = numpy.asarray(data["y"])

    random_state = rd(0, 4294967295)
    random_state = 0
    return sklearn.model_selection.train_test_split(
        X, y, test_size=1 - train_ratio, random_state=random_state
    )


def scale_features(X_train, X_test, low=0, upp=1):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(low, upp)).fit(
        numpy.vstack((X_train, X_test))
    )
    X_train_scale = minmax_scaler.transform(X_train)
    X_test_scale = minmax_scaler.transform(X_test)
    return X_train_scale, X_test_scale


def cross_entropy(y, y_hat):
    loss = 0
    for i in range(len(y)):
        loss += -(y[i] * math.log(y_hat[i]) + (1 - y[i]) * math.log(1 - y_hat[i]))
    return loss


def logreg_sgd(X, y, alpha=0.001, epochs=10000, eps=1e-4):
    # TODO: compute theta
    # alpha: step size
    # epochs: max epochs
    # eps: stop when the thetas between two epochs are all less than eps

    n, d = X.shape
    theta = numpy.zeros((d, 1))
    old_theta = numpy.zeros((d, 1))

    for _ in range(epochs):
        for i in range(n):
            y_hat = predict_prob(X[i], theta)
            grandient = numpy.dot(X[i].reshape(d, 1), (y_hat - y[i]).reshape(1, 1))
            theta = theta - alpha * grandient
        if (numpy.abs(theta - old_theta) >= eps).sum() == 0:
            break
        old_theta = copy.deepcopy(theta)
        if _ % 100 == 0:
            print(f"epoch: {_}, loss: {cross_entropy(y, predict_prob(X, theta))}")

    return theta


def predict_prob(X, theta):
    return 1.0 / (1 + numpy.exp(-numpy.dot(X, theta)))

def cmp(x, y):
    if x[0] != y[0]:
        return 1 if x[0] > y[0] else -1
    else:
        return 1 if x[1] > y[1] else -1

def plot_roc_curve(y_test, y_prob):
    # TODO: compute tpr and fpr of different thresholds
    tpr, fpr = [], []
    y_prob, y_test = zip(*sorted(zip(y_prob, y_test), key=lambda x: x[0]))
    y_prob, y_test = y_prob[::-1], y_test[::-1]
    fn, tn = 0, 0
    for i in y_test:
        if i:
            fn += 1
        else:
            tn += 1
    tp, fp = 0, 0
    for i in range(len(y_prob)):
        if y_test[i]:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / fn)
        fpr.append(fp / tn)

    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig("roc_curve.png")
    # calculate AUC
    fpr, tpr = zip(*sorted(zip(fpr, tpr), key=cmp_to_key(cmp)))
    nfpr, ntpr = [], []
    last = -1.0
    for i in range(len(fpr)):
        if fpr[i] == last:
            ntpr[-1] = tpr[i]
        else:
            nfpr.append(fpr[i])
            ntpr.append(tpr[i])
            last = fpr[i]
    print("Area Under ROC Curve: %f" % (sklearn.metrics.auc(nfpr, ntpr)))


def main(argv):
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=0.5)
    X_train_scale, X_test_scale = scale_features(X_train, X_test, 0, 1)

    theta = logreg_sgd(X_train_scale, y_train, epochs=1000)
    print("")
    print(theta)
    y_prob = predict_prob(X_train_scale, theta)
    threadhold = 0.5
    print(
        "Logreg train accuracy: %f"
        % (sklearn.metrics.accuracy_score(y_train, y_prob > threadhold))
    )
    print(
        "Logreg train precision: %f"
        % (sklearn.metrics.precision_score(y_train, y_prob > threadhold))
    )
    print(
        "Logreg train recall: %f"
        % (sklearn.metrics.recall_score(y_train, y_prob > threadhold))
    )
    y_prob = predict_prob(X_test_scale, theta)
    print(
        "Logreg test accuracy: %f"
        % (sklearn.metrics.accuracy_score(y_test, y_prob > threadhold))
    )
    print(
        "Logreg test precision: %f"
        % (sklearn.metrics.precision_score(y_test, y_prob > threadhold))
    )
    print(
        "Logreg test recall: %f"
        % (sklearn.metrics.recall_score(y_test, y_prob > threadhold))
    )
    plot_roc_curve(y_test.flatten(), y_prob.flatten())


if __name__ == "__main__":
    main(sys.argv)
    
