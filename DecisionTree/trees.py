# Machine Learning in Action by Peter Harrington.
# Manning publications.
# Chapter 3.

from math import log
import operator
import pickle

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
    baseEntropy = calcShannonEnt(_dataSet)
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
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(_classList):
    classCount = {}
    for vote in _classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = \
                        sorted(classCount.iteritems(), \
                        # Get the first item.
                        key = operator.itemgetter(1), \
                        reverse = True)
    return sortedClassCount[0][0]

def createTree(_dataSet, _labels):
    classList = [example[-1] for example in _dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(_dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(_dataSet)
    bestFeatLabel = _labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(_labels[bestFeat])
    featValues = [example[bestFeat] for example in _dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # Create a copy of the _labels list.
        subLabels = _labels[:]
        myTree[bestFeatLabel][value] = \
                        createTree(splitDataSet(_dataSet, \
                        bestFeat, value), \
                        subLabels)
    return myTree

def classify(_inputTree, _featLabels, _testVec):
    firstStr = list(_inputTree.keys())[0]
    secondDict = _inputTree[firstStr]
    featIndex = _featLabels.index(firstStr)
    for key in secondDict.keys():
        if _testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == "dict":
                classLabel = classify(secondDict[key], \
                                      _featLabels, _testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(_inputTree, _fileName):
    fw = open(_fileName, 'wb')
    pickle.dump(_inputTree, fw)
    fw.close()

def grabTree(_fileName):
    fr = open(_fileName, 'rb')
    return pickle.load(fr)
