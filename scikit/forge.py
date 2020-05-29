# Introduction to Machine Learning with Python
# Book by Andreas C. Müller and Sarah Guido
# Chapter 2

from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
import mglearn
import matplotlib.pyplot as plt
import numpy as np

def knnForgeEG():
    X, y = mglearn.datasets.make_forge()
    XTrain, XTest, yTrain, yTest = train_test_split(\
        X, y, random_state = 0)
    clf = KNeighborsClassifier(n_neighbors = 3)
    clf.fit(XTrain, yTrain)
    print("Test set predictions: {}".format(clf.predict(XTest)))
    print("Test set accuracy: {:.2f}".format(clf.score(\
        XTest, yTest)))
    fig, axes = plt.subplots(1, 3, figsize = (10, 3))
    for nNeighbors, ax in zip([1, 3, 9], axes):
        clf = KNeighborsClassifier(n_neighbors = nNeighbors)\
              .fit(X, y)
        mglearn.plots.plot_2d_separator(clf, X, fill = True, \
                                        eps = .5, ax = ax,
                                        alpha = .4)
        mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax = ax)
        ax.set_title("{} neighbor(s)".format(nNeighbors))
        ax.set_xlabel("feature 0")
        ax.set_ylabel("feature 1")
    axes[0].legend(loc = 3)
    plt.show()
    
def knnCancerEG():
    cancer = load_breast_cancer()
    #stratify parameter will preserve the proportion.
    XTrain, XTest, yTrain, yTest = train_test_split(\
        cancer.data, cancer.target, stratify = cancer.target,
        random_state = 66)
    trainingAccuracy = []
    testAccuracy = []
    neighborSettings = range(1, 11)
    for nNeighbors in neighborSettings:
        clf = KNeighborsClassifier(n_neighbors = nNeighbors)
        clf.fit(XTrain, yTrain)
        trainingAccuracy.append(clf.score(XTrain, yTrain))
        testAccuracy.append(clf.score(XTest, yTest))
    plt.plot(neighborSettings, trainingAccuracy, label = \
             "Training Accuracy")
    plt.plot(neighborSettings, testAccuracy, label = \
             "Test Accuracy")
    plt.xlabel("nNeighbors")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

def knnWaveEG():
    X, y = mglearn.datasets.make_wave(n_samples = 40)
    XTrain, XTest, yTrain, yTest = train_test_split(\
        X, y, random_state = 0)
    reg = KNeighborsRegressor(n_neighbors = 3)
    reg.fit(XTrain, yTrain)
    print("Test set predictions:\n{}".format(reg.predict(XTest)))
    print("Test set R^2: {:.2f}".format(reg.score(XTest, yTest)))
    fig, axes = plt.subplots(1, 3, figsize = (15, 4))
    # reshape(-1, 1), -1 -> unknown dimension.
    line = np.linspace(-3, 3, 1000).reshape(-1, 1)
    for nNeighbors, ax in zip([1, 3, 8], axes):
        reg = KNeighborsRegressor(n_neighbors = nNeighbors)
        reg.fit(XTrain, yTrain)
        ax.plot(line, reg.predict(line))
        ax.plot(XTrain, yTrain, '^', c = mglearn.cm2(0), \
                markersize = 8)
        ax.plot(XTest, yTest, 'v', c = mglearn.cm2(1), \
                markersize = 8)
        ax.set_title(("{} neighbor(s)\n train score: {:.2f}" \
                      "test score: {:.2f}").format(nNeighbors, \
                                    reg.score(XTrain, yTrain), \
                                    reg.score(XTest, yTest)))
        ax.set_xlabel("Feature")
        ax.set_ylabel("Target")
        axes[0].legend(["Model predictions", \
                        "Training data/target", \
                        "Test data/target"], \
                        loc = "best")
    plt.show()
    
# e.g. Classification data.
def classificationEG():
    X, y = mglearn.datasets.make_forge()
    mglearn.discrete_scatter(X[:,0], X[:, 1], y)
    plt.legend(["Class 0", "Class 1"], loc = 4)
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    print("X.shape: {}".format(X.shape))
    #mglearn.plots.plot_knn_classification(n_neighbors = 1)
    mglearn.plots.plot_knn_classification(n_neighbors = 3)
    plt.show()
    
#e.g. Regression data.
def regressionEG():
    X,y = mglearn.datasets.make_wave(n_samples = 40)
    plt.plot(X, y, 'o')
    plt.ylim(-3, 3)
    plt.xlabel("Feature")
    plt.ylabel("Target")
    #mglearn.plots.plot_knn_regression(n_neighbors = 1)
    mglearn.plots.plot_knn_regression(n_neighbors = 3)
    plt.show()

def cancerEG():
    cancer = load_breast_cancer() # Bunch Object.
    print("cancer.keys(): \n{}".format(cancer.keys()))
    print("Shape of cancer data: {}".format(cancer.data.shape))
    print("Sample counts per class:\n{}".format(\
        {n: v for n, v in zip(cancer.target_names, \
                              np.bincount(cancer.target))}))
    print("Feature Names:\n{}".format(cancer.feature_names))
    #print("Data Description:\n{}".format(cancer.DESCR))
    #print("Data Description:\n{}".format(cancer["DESCR"]))

def bostonEG():
    boston = load_boston() # Bunch Object.
    print("Data shape: {}".format(boston.data.shape))
    #print("Data shape: {}".format(boston["data"].shape))
    #print("Data Description: {}".format(boston.DESCR))
    #print("Data description: {}".format(boston["DESCR"]))
    X, y = mglearn.datasets.load_extended_boston()
    print("X.shape: {}".format(X.shape))
    
    

def main():
    #classificationEG()
    #regressionEG()
    #cancerEG()
    #bostonEG()
    #knnForgeEG()
    #knnCancerEG()
    knnWaveEG()
main()
