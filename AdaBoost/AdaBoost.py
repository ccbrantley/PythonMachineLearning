# Machine Learning in Action by Peter Harrington
# Mannings Publications
# Chapter 7
from numpy import *
import matplotlib.pyplot as plt
def loadSimpData():
    dataMat = matrix([[1.0, 2.1],
                      [2.0, 1.1],
                      [1.3, 1.0],
                      [1.0, 1.0],
                      [2.0, 1.0]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

def loadDataSet(_fileName):
    numFeat = len(open(_fileName).readline().split("\t"))
    dataMat = []
    labelMat = []
    fr = open(_fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split("\t")
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def stumpClassify(_dataMatrix, _dimen, \
                  _threshVal, _threshIneq):
    retArray = ones((shape(_dataMatrix)[0],1))
    if _threshIneq == "lt":
        retArray[_dataMatrix[:, _dimen] <= _threshVal] = -1.0
    else:
        retArray[_dataMatrix[:, _dimen] > _threshVal] = -1.0
    return retArray

def buildStump(_dataArr, _classLabels, _D):
    dataMatrix = mat(_dataArr)
    labelMat = mat(_classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = mat(zeros((m,1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax-rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ["lt", "gt"]:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, \
                                              i, threshVal, \
                                              inequal)
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = _D.T * errArr

                #print(("split: dim {}, thresh {:.2f}, thresh ineqal: {} " \
                #       "the weighted error is {:.3f}").format(i, threshVal, \
                #                      inequal, array(weightedError[0,0])))
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump["dim"] = i
                    bestStump["thresh"] = threshVal
                    bestStump["ineq"] = inequal
    return bestStump, minError, bestClassEst

def adaBoostTrainDS(_dataArr, _classLabels, _numIt = 40):
    weakClassArr = []
    m = shape(_dataArr)[0]
    D = mat(ones((m, 1)) / m)
    aggClassEst = mat(zeros((m, 1)))
    for i in range(_numIt):
        bestStump, error, classEst = buildStump(_dataArr, \
                                            _classLabels, D)
        print("D: ", D.T)
        # 1e-16 -> prevents divide by zero.
        alpha = float(0.5 * log((1.0 - error) / \
                                max(error, 1e-16)))
        bestStump["alpha"] = alpha
        weakClassArr.append(bestStump)
        print("Class Estimate: ", classEst.T)
        # Correctly predicted -> ++ -- -> -.
        # Incorrectly predicted -> -+ +- -> +.
        expon = multiply(-1 * alpha * mat(_classLabels).T, \
                         classEst)
        D = multiply(D, exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
        print("Aggregate Class Estimate: ", aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != \
                             mat(_classLabels).T, ones((m,1)))
        errorRate = aggErrors.sum() / m
        print("Total Error: {:.2f}\n".format(errorRate))
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst

def adaClassify(_datToClass, _classifierArr):
    dataMatrix = mat(_datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(_classifierArr)):
        classEst = stumpClassify(dataMatrix, \
                                 _classifierArr[i]["dim"], \
                                 _classifierArr[i]["thresh"], \
                                 _classifierArr[i]["ineq"])
        aggClassEst += _classifierArr[i]["alpha"] * classEst
        print("Aggregate Class Estimate: ", aggClassEst)
    return sign(aggClassEst)

def plotROC(_predStrengths, _classLabels):
    cur = ( 1.0, 1.0)
    ySum = 0.0
    numPosClas = sum(array(_classLabels) == 1.0)
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(_classLabels) - numPosClas)
    sortedIndicies = _predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if _classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - delX], \
                [cur[1], cur[1] - delY], c = "b")
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], "b--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve for AdaBoost Horse Colic " \
              "Detection System")
    ax.axis([0, 1, 0, 1])
    print("The are under the curve is: ", ySum * xStep)
    plt.show()

def main():
    # e.g. 1
    datMat, classLabels = loadSimpData()
    D = mat(ones((5, 1))/5)
    bestStump, minError, bestClassEst = \
               buildStump(datMat, classLabels, D)
    print("Best Stump: {}".format(bestStump))
    print("Min Error: {:.2f}".format(float(minError)))
    print("Best Class Estimate: \n{}".format(bestClassEst))
    datMat, classLabels = loadSimpData()
    classifier, aggClassEstTrash = adaBoostTrainDS(datMat, classLabels, 30)
    print("Classifier: \n", classifier)

    # Classification.
    classification = adaClassify([0, 0], classifier)
    print("Classification [0, 0]: ", classification)
    classification2 = adaClassify([[5, 5], [0, 0]], classifier)
    print("Classification [5, 5]: ", classification2[0])
    print("Classification [0, 0]: ", classification2[1])

    #e.g. 2
    trainingFilePath = "../LogisticRegression/horseColicTraining.txt"
    dataArr, labelArr = loadDataSet(trainingFilePath)
    classifierArray, aggClassEst = adaBoostTrainDS(dataArr, labelArr, 10)
    testFilePath = \
        "../LogisticRegression/horseColicTest.txt"
    testArr, testLabelArr = loadDataSet(testFilePath)
    prediction10 = adaClassify(testArr, classifierArray)
    errArr = mat(ones((67, 1)))
    errorCount = errArr[prediction10 != \
                        mat(testLabelArr).T].sum()
    print("Error rate: {}".format(errorCount / 67))
    plotROC(aggClassEst.T, labelArr)
main()
