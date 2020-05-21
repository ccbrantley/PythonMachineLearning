# Machine Learning in Action by Peter Harrington
# Manning publications
# Chapter 3
from math import log

def createDataSet():
    dataSet = [[1, 1, "yes"],
               [1, 1, "yes"],
               [1, 0, "no"],
               [0, 1, "no"],
               [0, 1, "no"]]
    labels = ["no surfacing", "flippers"]
    return dataSet, labels

def splitDataSet(_dataSet, _axis, _value):
    retDataSet = []
    for featVec in _dataSet:
        if featVec[_axis] == _value:
            reducedFeatVec = featVec[:_axis]
            reducedFeatVec.extend(featVec[_axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def calcShannonEnt(_dataSet):
    numEntries = len(_dataSet)
    labelCounts = {}
    for featVec in _dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def chooseBestFeatureToSplit(_dataSet):
    numFeatures = len(_dataSet[0]) - 1
    baseEntroppy = calcShannonEnt(_dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in _dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(_dataSet, i, value)
            prob = len(subDataSet)/float(len(_dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestinfoGain = infoGain
            bestFeature = i
    return bestFeature
            


