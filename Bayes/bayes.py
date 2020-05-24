# Machine Learning in Action by Peter Harrington
# Mannings Publications
# Chapter 4

from numpy import *
import re
import operator
import feedparser

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

def textParse(_bigString):
    listOfTokens = re.split(r'\W', _bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):
        wordList  = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    
    # Application of Hold-out cross validation.
    for i in range(10):
        randIndex = int(random.uniform(0, 5))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    
    # Training.
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, \
                                       docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), \
                               array(trainClasses))
    errorCount = 0
    
    # Testing.
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, \
                                    docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) \
           != classList[docIndex]:
            errorCount += 1
            print("Classification Error: ", docList[docIndex])
    print( "The error rate is: ", float(errorCount) / \
                                       len(testSet))

def calcMostFreq(_vocabList, _fullText):
    freqDict = {}
    for token in _vocabList:
        freqDict[token] = _fullText.count(token)
    sortedFreq = sorted(freqDict.items(), \
                        # Sort by first item.
                        key = operator.itemgetter(1), \
                        reverse = True)
    return sortedFreq[:30]

def localWords(_feed1, _feed0):
    docList = []
    classList = []
    fullText = []
    minLen = min(len(_feed1['entries']), len(_feed0['entries']))
    for i in range(minLen):
        wordList = textParse(_feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) # 1 -> houston
        wordList = textParse(_feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0) # 0 -> New York
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = list(range(2 * minLen))
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, \
                                         docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), \
                               array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, \
                                      docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) \
           != classList[docIndex]:
            errorCount += 1
    print("The error rate is: ", float(errorCount) / len(testSet))
    return vocabList, p0V, p1V

def getTopWords(_houston, _newYork):
    vocabList, p0V, p1V = localWords(_houston, _newYork)
    tophouston = []
    topNewYork = []
    # 1 -> houston
    # 0 -> New York
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            tophouston.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNewYork.append((vocabList[i], p1V[i]))
    sortedhouston = sorted(tophouston, \
                          key = lambda pair: pair[1], \
                          reverse = True)
    sortedNewYork = sorted(topNewYork, \
                           key = lambda pair: pair[1], \
                           reverse = True)
    print("\nhouston Top Words")
    count = 0
    for item in sortedhouston:
        print(item[0])
        if count == 10:
            break
        count += 1

    count = 0 
    print("\nNew York Top Words")
    for item in sortedNewYork:
        print(item[0])
        if count == 10:
            break
        count += 1
        
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
          classifyNB(testVec, p0V, p1V, pAB), "\n")

    print("Spam Email Classifier Error Rate")
    for x in range(5):
        spamTest()

    # RSS feed analysis of political identity.
    print("\n\nRSS feed analysis of political identity.")
    houstonRSS = \
    "https://houston.craigslist.org/search/pol?format=rss"
    newYorkRSS = \
    "https://newyork.craigslist.org/search/pol?format=rss"
    houston = feedparser.parse(houstonRSS)
    newYork = feedparser.parse(newYorkRSS)
    getTopWords(houston, newYork)
main()
