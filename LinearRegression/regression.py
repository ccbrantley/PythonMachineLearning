# Machine Learning in Action by Peter Harrington.
# Manning publications.
# Chapter 8.

from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(_fileName):
    numFeat = len(open(_fileName).readline().split("\t")) - 1
    dataMat = []
    labelMat = []
    fr = open(_fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split("\t")
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegres(_xArr, _yArr):
    xMat = mat(_xArr)
    yMat = mat(_yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print("This matrix is non-invertible, no inverse.")
        return
    ws = xTx.I * xMat.T * yMat
    return ws

def lwlr(_testPoint, _xArr, _yArr, _k = 1.0):
    xMat = mat(_xArr)
    yMat = mat(_yArr).T
    m = shape(xMat)[0]
    # returns a 2-D diagonal array
    weights = mat(eye((m)))
    for j in range(m):
        diffMat = _testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat * diffMat.T / \
                            (-2.0 * _k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is non-invertible, no inverse.")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return _testPoint * ws

def lwlrTest(_testArr, _xArr, _yArr, _k = 1.0):
    m = shape(_testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(_testArr[i], _xArr, _yArr, _k)
    return yHat
        
def eg1():
    xArr, yArr = loadDataSet('ex0.txt')
    print(xArr[0: 2])
    ws = standRegres(xArr, yArr)
    print(ws)
    xMat = mat(xArr)
    yMat = mat(yArr)
    yHat = xMat * ws
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # flatten() gets copy of array in 1d.
    ax.scatter(xMat[:, 1].flatten().A[0], \
               yMat.T[:, 0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    yHat = xMat * ws
    corrCoef = corrcoef(yHat.T, yMat)
    print("Correlation Coefficient:\n{}".format(corrCoef))
    plt.show()

def eg2():
    xArr, yArr = loadDataSet('ex0.txt')
    print("Original Target Value: {:.2f}".format(yArr[0]))
    tv1 = lwlr(xArr[0], xArr, yArr, 1.0)
    tv2 = lwlr(xArr[0], xArr, yArr, .001)
    print("Target Value when K = 1.0: {:.2f}".format(tv1[0, 0]))
    print("Target value when k = .001: {:.2f}".format(tv2[0, 0]))
    yHat = lwlrTest(xArr, xArr, yArr, 0.003)
    xMat = mat(xArr)
    # Sorts by first column.
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:, 0, :]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[srtInd])
    # flatten() gets copy of array in 1d.
    ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr)\
               .T.flatten().A[0], s = 2, c = "red")
    plt.show()
    
def main():
    #eg1()
    eg2()
main()
