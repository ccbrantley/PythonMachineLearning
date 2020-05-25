# Machine Learning in Action by Peter Harrington
# Manning Publication
# Chapter 5

from numpy import *
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), \
                        float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(_inX):
    # Exponential value with the base set to e.
    return 1.0 / (1 + exp(-_inX))

def gradAscent(_dataMatIn, _classLabels):
    dataMatrix = mat(_dataMatIn)
    labelMatrix = mat(_classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = .001
    maxCycles = 500
    # Shape of the new matrix (row, column)
    # (m,n) * (n,1)
    weights = ones((n,1))
    for k in range(maxCycles):
        # This will return a column vector.
        h = sigmoid(dataMatrix * weights)
        error = (labelMatrix - h)
        weights = weights + alpha * \
                  dataMatrix.transpose() * error
    return weights

def plotBestFit(_wei):
    weights = mat(_wei).getA() # Get Array
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    xcord2 = []
    ycord1 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            # 0 -> weight, 1 -> x, 2 -> y.
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
    ax.scatter(xcord2, ycord2, s = 30, c = 'green')
    x = arange(-3.0, 3.0, 0.1)
    # Indexes may change depending on matrix/array/etc.
    y = (- weights[0][0] - weights[0][1] * x) / weights[0][2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def stocGradAscent0(_dataArray, _classLabels, _numIter = 150):
    m,n = shape(_dataArray)
    weights = ones(n)
    for j in range(_numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # alpha approaches 0 as iterations increase.
            # (j + i) avoids strictly decreasing.
            alpha = 4 / (1.0 + j + i) + .01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(_dataArray[randIndex] * weights))
            error = _classLabels[randIndex] - h
            weights = weights + alpha * \
                      error * _dataArray[randIndex]
            del(dataIndex[randIndex])
    return weights

def classifyVector(_inX, _weights):
    prob = sigmoid(sum(_inX * _weights))
    if prob > .5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open("horseColicTraining.txt")
    frTest = open("horseColicTest.txt")
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split("\t")
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent0(array(trainingSet), \
                                   trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split("\t")
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) \
           != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("The error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("After %d iterations the average error rate is: %f" \
          % (numTests, errorSum / float(numTests)))

def main():
    #e.g. 1
    #dataArr, labels = loadDataSet()
    #weights = mat(stocGradAscent0(array(dataArr), labels))
    #plotBestFit(weights)

    #e.g. 2
    multiTest()
main()
