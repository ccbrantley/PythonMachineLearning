# Introduction to Machine Learning with Python
# Book by Andreas C. Müller and Sarah Guido
# Chapter 2

from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
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


def leastSquaresEG():
    X, y = mglearn.datasets.make_wave(n_samples = 60)
    XTrain, XTest, yTrain, yTest = train_test_split(\
        X, y, random_state = 42)
    lr = LinearRegression().fit(XTrain, yTrain)
    print("lr.coef_: {:,.2f}".format(lr.coef_[0]))
    print("lr.intercept_: {:,.2f}".format(lr.intercept_))
    print("Training set score: {:,.2f}".format(\
        lr.score(XTrain, yTrain)))
    print("Test set score: {:,.2f}".format(\
        lr.score(XTest, yTest)))
    X, y = mglearn.datasets.load_extended_boston()
    XTrain, XTest, yTrain, yTest = train_test_split(\
        X, y, random_state = 0)
    lr = LinearRegression().fit(XTrain, yTrain)
    print("Training set score: {:,.2f}".format(\
        lr.score(XTrain, yTrain)))
    print("Test set score: {:,.2f}".format(\
        lr.score(XTest, yTest)))

def ridgeRegression():
    X, y = mglearn.datasets.load_extended_boston()
    XTrain, XTest, yTrain, yTest = train_test_split(\
        X, y, random_state = 0)
    ridge = Ridge().fit(XTrain, yTrain)
    print("\nTraining set score (alpha = 1): {:,.2f}".format(\
        ridge.score(XTrain, yTrain)))
    print("Test set score (alpha = 1): {:,.2f}".format(\
        ridge.score(XTest, yTest)))
    # Increasing alpha -> coefficients -> zero
    ridge10 = Ridge(alpha = 10).fit(XTrain, yTrain)
    print("\nTraining set score (alpha = 10): {:,.2f}".format(\
        ridge10.score(XTrain, yTrain)))
    print("Test set score (alpha = 10): {:,.2f}".format(\
        ridge10.score(XTest, yTest)))
    # Increased the alpha attempting to underfit the model.
    # Discovered we should instead be overfitting.
    ridge01 = Ridge(alpha = .1).fit(XTrain, yTrain)
    print("\nTraining set score (alpha = .1): {:,.2f}".format( \
        ridge01.score(XTrain, yTrain)))
    print("Test set score (alpha = .1): {:,.2f}".format( \
        ridge01.score(XTest, yTest)))
    
    lr = LinearRegression().fit(XTrain, yTrain)
    #plt.plot(ridge.coef_, 's', label = "Ridge alpha = 1")
    #plt.plot(ridge10.coef_, '^', label = "Ridge alpha = 10")
    #plt.plot(ridge01.coef_, 'v', label = "Ridge alpha = .1")
    #plt.plot(lr.coef_, 'o', label = "Linear Regression")
    #plt.xlabel("Coefficient index")
    #plt.ylabel("Coefficient magnitude")
    #plt.hlines(0, 0, len(lr.coef_))
    #plt.ylim(-25, 25)
    #plt.legend()
    #plt.show()
    mglearn.plots.plot_ridge_n_samples()
    plt.show()

def lassoRegression():
    X, y = mglearn.datasets.load_extended_boston()
    XTrain, XTest, yTrain, yTest = train_test_split(\
        X, y, random_state = 0)
    lasso = Lasso().fit(XTrain, yTrain)
    print("\nTraining set score (alpha = 1): {:,.2f}".format(\
        lasso.score(XTrain, yTrain)))
    print("Test set score (alpha = 1): {:,.2f}".format(\
        lasso.score(XTest, yTest)))
    print("Number of features used (alpha = 1): {}".format(\
        np.sum(lasso.coef_ != 0)))
    lasso001 = Lasso(alpha = .01, max_iter = 100000).fit(\
        XTrain, yTrain)
    print("\nTraining set score (alpha = .01): {:,.2f}".format(\
        lasso001.score(XTrain, yTrain)))
    print("Test set score (alpha = .01): {:,.2f}".format(\
        lasso001.score(XTest, yTest)))
    print("Number of features used (alpha = .01): {}".format(\
        np.sum(lasso001.coef_ != 0)))
    lasso00001 = Lasso(alpha = .0001, max_iter = 100000).fit(\
        XTrain, yTrain)
    print("\nTraining set score (alpha = .0001): {:,.2f}".format(\
        lasso00001.score(XTrain, yTrain)))
    print("Test set score (alpha = .0001): {:,.2f}".format(\
        lasso00001.score(XTest, yTest)))
    print("Number of features used (alpha = .0001): {}".format(\
        np.sum(lasso00001.coef_ != 0)))

    plt.plot(lasso.coef_, 's', label = "Lasso alpha = 1")
    plt.plot(lasso001.coef_, '^', label = "Lasso alpha = .01")
    plt.plot(lasso00001.coef_, 'v', label = "Lasso alpha = .0001")
    ridge01 = Ridge(alpha = .1).fit(XTrain, yTrain)
    plt.plot(ridge01.coef_, 'o', label = "Ridge alpha = .1")
    plt.legend(ncol = 2, loc = (0, 1.05))
    plt.ylim(-25, 25)
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient magnitude")
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
    #knnWaveEG()
    #leastSquaresEG()
    #ridgeRegression()
    lassoRegression()
main()
