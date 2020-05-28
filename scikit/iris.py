# Introduction to Machine Learning with Python
# Book by Andreas C. Müller and Sarah Guido
# Chapter 1

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt
import mglearn
import numpy as np
import pandas as pd

def main():
    irisData = load_iris()
    print("Keys of irisData: \n{}".format(irisData.keys()))
    print(irisData['DESCR'][:200], "\n")
    print("Target names: {}".format(irisData["target_names"]))
    print("Feature names: {}".format(irisData["feature_names"]))
    print("Type of data: {}".format(type(irisData["data"])))
    print("Shape of data: {}".format(irisData["data"].shape))
    print("First five columns of data: \n{}"\
          .format(irisData['data'][:5]))
    print("Type of target: {}".format(type(irisData["target"])))
    print("Shape of target: {}".format(irisData["target"].shape))
    print("Target: \n{}".format(irisData["target"]))
    print("Target meanings:\n{}".format(irisData["target_names"]))

    # Capital X -> matrix, lower y -> array
    # random_state = 0, fixed seed -> deterministic
    XTrain, XTest, yTrain, yTest = train_test_split(\
        irisData["data"], irisData["target"], random_state = 0)
    print("XTrain shape: {}".format(XTrain.shape))
    print("yTrain shape: {}".format(yTrain.shape))
    print("XTest shape: {}".format(XTest.shape))
    print("yTest shape: {}".format(yTest.shape))

    # Create dataframe from data in XTrain.
    # Label the columns using the features.
    print("Feature names: {}".format(irisData.feature_names))
    irisDataFrame = pd.DataFrame(XTrain, columns = \
                                 irisData.feature_names)
    # Create a scatter matrix, color by yTrain
    plot = pd.plotting.scatter_matrix(irisDataFrame, c = yTrain, \
                    figsize = (15, 15), \
                    marker = 'o', alpha = .8, \
                    hist_kwds = {"bins" : 20}, s = 60, \
                    cmap = mglearn.cm3)
    # Shows plots.
    #plt.show()
    knn = KNeighborsClassifier(n_neighbors = 1)
    print(knn.fit(XTrain, yTrain))
    # Scikit requests 2-d data.
    xNew = np.array([[5, 2.9, 1, .2]])
    print("xNew shape: {}".format(xNew.shape))
    prediction = knn.predict(xNew)
    print("Prediction: {}".format(prediction))
    print("Predicted target name: {}".format(\
        irisData["target_names"][prediction]))
    yPred  = knn.predict(XTest)
    print("Test set predictions:\n{}".format(yPred))
    print("Test set score: {:.2f}".format(\
        np.mean(yPred == yTest)))
    print("Test set score(knn method): {:.2f}".format(\
        knn.score(XTest, yTest)))

    print("\nTest")
    print("{0:d} {1:d} ".format(12, 31))
    print("{1:d} {0:d} ".format(12, 31))

    #Summary
    XTrain, XTest, yTrain, yTest = train_test_split(\
        irisData["data"], irisData["target"], \
        random_state = 0)
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(XTrain, yTrain)
    print("Test set score: {:.2f}".format(knn.score(\
        XTest, yTest)))
    
                             
main()
