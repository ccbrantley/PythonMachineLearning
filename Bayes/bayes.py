# Machine Learning in Action by Peter Harrington
# Mannings Publications
# Chapter 4

from numpy import *
    
def loadDataSet():
    postingList = [["my", "dog", "has", "flea", \
                    "problems", "help", "him", "please"],
                   ["maybe", "not", "take", "him", \
                    "to", "dog", "park", "stupid"],
                   ["my", "dalmation", "is", "so", "cute", \
                    "I", "love", "him"],
                   ["stop", "posting", "stupid", "worthless", \
                    "garbage"],
                   ["mr", "licks", "ate", "my", "steak", "how", \
                    "to", "stop", "him"],
                   ["quit", "buying", "worthless", "dog", "food", \
                    "stupid"]]
    classVec = [0,1,0,1,0,1] # 1 -> abusive, 0 -> non-abusve
    return postingList, classVec

def createVocabList(_dataSet):
    vocabSet = set([])
    for document in _dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(_vocabList, _inputSet):
    returnVec = [0] * len(_vocabList)
    for word in _inputSet:
        if word in _vocabList:
            returnVec[_vocabList.index(word)] = 1
        else:
            print ("The word: %s is not in my vocabulary!"  \
                   % word)
    return returnVec

def bagOfWords2VecMN(_vocabList, _inputSet):
    returnVec = [0] * len(_vocabList)
    for word in _inputSet:
        if word in _vocabList:
            returnVec[_vocabList.index(word)] += 1
    return returnVec
    
# independently likely, conditional independence
def trainNB0(_trainMatrix, _trainCategory):
    numTrainDocs = len(_trainMatrix)
    numWords = len(_trainMatrix[0])
    pAbusive = sum(_trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if _trainCategory[i] == 1:
            p1Num += _trainMatrix[i]
            p1Denom += sum(_trainMatrix[i])
        else:
            p0Num += _trainMatrix[i]
            p0Denom += sum(_trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(_vec2Classify, _p0Vec, _p1Vec, _pClass1):
    p1 = sum(_vec2Classify * _p1Vec) + log(_pClass1)
    p0 = sum(_vec2Classify * _p0Vec) + log(1.0 - _pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def main():
    posts, labels = loadDataSet()
    vocabList = createVocabList(posts)
    trainMat = []
    for record in posts:
        trainMat.append(setOfWords2Vec(vocabList, record))
    p0V, p1V, pAB = trainNB0(array(trainMat), array(labels))
    testEntry = ["love", "my", "dalmation"]
    testVec = array(setOfWords2Vec(vocabList, testEntry))
    print(testEntry, "was classified as: ", \
          classifyNB(testVec, p0V, p1V, pAB))
    testEntry2 = ["stupid", "garbage"]
    testVec = array(setOfWords2Vec(vocabList, testEntry2))
    print(testEntry2, "was classified as: ", \
          classifyNB(testVec, p0V, p1V, pAB))
main()
