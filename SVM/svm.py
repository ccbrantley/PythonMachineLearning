# Machine Learning in Action by Peter Harrington
# Manning Publication
# Chapter 6
# Sequential minimal optimization

from numpy import *
from OptStruct import OptStruct
from os import listdir

def loadDataSet(_fileName):
    dataMat = []
    labelMat = []
    fr = open(_fileName)
    for line in fr.readlines():
        lineArr = line.strip().split("\t")
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

# index of first alpha, total number of alphas
def selectJRand(_i, _m):
    j = _i
    while (j == _i):
        j = int(random.uniform(0, _m))
    return j

def clipAlpha(_aj, _h, _l):
    if _aj > _h:
        _aj = _h
    if _l > _aj:
        _aj = _l
    return _aj

def smoSimple(_dataMatIn, _classLabels, _C, _toler, _maxIter):
    dataMatrix = mat(_dataMatIn)
    labelMat = mat(_classLabels).transpose()
    b = 0
    m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    iterCount = 0
    while (iterCount < _maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            # .T returns the transpose.
            fXi = float(multiply(alphas, labelMat).T * \
                  (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i] * Ei < -_toler) and \
                (alphas[i] < _C)) or \
                ((labelMat[i] * Ei > _toler) and \
                 (alphas[i] > 0)):
                j = selectJRand(i, m)
                fXj = float(multiply(alphas, labelMat).T * \
                            (dataMatrix * dataMatrix[j, :].T) \
                            ) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(_C, _C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - _C)
                    H = min(_C, alphas[j] + alphas[i])
                if L == H:
                    print("L == H")
                    continue
                # Optimal amount to change.
                eta = 2.0 * dataMatrix[i, :] * \
                      dataMatrix[j, :].T - dataMatrix[i, :] * \
                      dataMatrix[i, :].T - dataMatrix[j, :] * \
                      dataMatrix[j, :].T
                if eta >= 0:
                    print("eta >= 0")
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < .00001):
                    print("j is not moving enough.")
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * \
                             (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - \
                        alphaIold) * dataMatrix[i, :] * \
                        dataMatrix[i, :].T - labelMat[j] * \
                        (alphas[j] - alphaJold) * \
                        dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - \
                        alphaIold) * dataMatrix[i, :] * \
                        dataMatrix[j, :].T - labelMat[j] * \
                        (alphas[j] - alphaJold) * \
                        dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (_C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (_C > alphas[j]):
                    b = b2
                else:
                    b = (b1  + b2) / 2.0
                alphaPairsChanged += 1
                print("Iteration: %d i:%d, pairs changed %d" \
                      % (iterCount, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iterCount += 1
        else:
            iterCount = 0
        print("Iteration Number: %d" % iterCount)
    return b, alphas  
       
def calcEk(_oS, _k):
    fXk = float(multiply(_oS.alphas, _oS.labelMat).T * \
                _oS.k[:, _k] + _oS.b)
    Ek = fXk - float(_oS.labelMat[_k])
    return Ek

def selectJ(_i, _oS, _Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    _oS.eCache[_i] = [1,_Ei]
    # .A returns the array.
    validEcacheList = nonzero(_oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == _i:
                continue
            Ek = calcEk(_oS, k)
            deltaE = abs(_Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJRand(_i, _oS.m)
        Ej = calcEk(_oS, j)
    return j, Ej

def updateEk(_oS, _k):
    Ek = calcEk(_oS, _k)
    _oS.eCache[_k] = [1, Ek]

def innerL(_i, _oS):
    Ei = calcEk(_oS, _i)
    if ((_oS.labelMat[_i] * Ei < -_oS.tol) and \
        (_oS.alphas[_i] < _oS.C)) or \
        ((_oS.labelMat[_i] * Ei > _oS.tol) and \
         (_oS.alphas[_i] > 0)):
        j, Ej = selectJ(_i, _oS, Ei)
        alphaIold = _oS.alphas[_i].copy()
        alphaJold = _oS.alphas[j].copy()
        if (_oS.labelMat[_i] != _oS.labelMat[j]):
            L = max(0, _oS.alphas[j] - _oS.alphas[_i] - _oS.C)
            H = min(_oS.C, _oS.C + _oS.alphas[j] - \
                    _oS.alphas[_i])
        else:
            L = max(0, _oS.alphas[j] + _oS.alphas[_i] - _oS.C)
            H = min(_oS.C, _oS.alphas[j] + _oS.alphas[_i])
        if L == H:
            print("L == H")
            return 0
        eta = 2.0 * _oS.k[_i, j] - _oS.k[_i, _i] - _oS.k[j, j]
        if eta >= 0:
            print("eta >= 0")
            return 0
        _oS.alphas[j] -= _oS.labelMat[j] * (Ei - Ej) / eta
        _oS.alphas[j] = clipAlpha(_oS.alphas[j], H, L)
        updateEk(_oS, j)
        if (abs(_oS.alphas[j] - alphaJold) < 0.00001):
            print("j is not moving enough.")
            return 0
        _oS.alphas[_i] += _oS.labelMat[j] * _oS.labelMat[_i] * \
                         (alphaJold - _oS.alphas[j])
        updateEk(_oS, _i)
        
        b1 = _oS.b - Ei - _oS.labelMat[_i] * \
             (_oS.alphas[_i] - alphaIold) * _oS.k[_i, _i] - \
             _oS.labelMat[j] * (_oS.alphas[j]-alphaJold) * \
             _oS.k[_i,j]

        b2 = _oS.b - Ej - _oS.labelMat[_i]* \
             (_oS.alphas[_i] - alphaIold) * _oS.k[_i,j]- \
             _oS.labelMat[j] * (_oS.alphas[j] - alphaJold) * \
             _oS.k[j,j]

        
        if (0 < _oS.alphas[_i]) and \
           (_oS.C > _oS.alphas[_i]):
            _oS.b = b1
        elif (0 < _oS.alphas[j]) and \
             (_oS.C > _oS.alphas[j]):
            _oS.b = b2
        else:
            _oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def smoP(_dataMatIn, _classLabels, _C, _toler, _maxIter, \
         _kTup = ("lin", 0)):
    _oS = OptStruct(mat(_dataMatIn), \
                    mat(_classLabels).transpose(), \
                    _C, _toler, _kTup)
    iterCount = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iterCount < _maxIter) and \
          ((alphaPairsChanged < 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(_oS.m):
                alphaPairsChanged += innerL(i, _oS)
            print("fullSet, iter: %d i:%d, pairs changed %d" \
                  % (iterCount, i, alphaPairsChanged))
            iterCount += 1
        else:
            nonBoundIs = nonzero((_oS.alphas.A >0) * \
                                 (_oS.alphas.A < _C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, _oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" \
                      (iterCount, i, alphaPairsChanged))
            iterCount += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iterCount)
    return _oS.b, _oS.alphas
       
def calcWs(_alphas, _dataArr, _classLabels):
    X = mat(_dataArr)
    labelMat = mat(_classLabels).transpose()
    m, n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(_alphas[i] * labelMat[i], X[i, :].T)
    return w

def kernelTrans(_X, _A, _kTup):
    m,n = shape(_X)
    K = mat(zeros((m, 1)))
    if _kTup[0] == "lin":
        K = _X * _A.T
    elif _kTup[0] == "rbf":
        for j in range(m):
            deltaRow = _X[j, :] - _A
            K[j] = deltaRow * deltaRow.T
        # exponential value, base -> e. 
        K = exp(K / (-1 * _kTup[1]**2))
    else:
        raise NameError("Houston We Have A Problem.", \
                        "That Kernel is not recognized.")
    return K
        
def testRbf(_k1 = 1.1):
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, \
                     10000, ("rbf", _k1))
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d support vectors" % \
          shape(sVs)[0])
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], \
                                    ("rbf", _k1))
        predict = kernelEval.T * multiply(labelSV, \
                                    alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("The training error rate is: %f" % \
         (float(errorCount) / m))
    dataArr, labelArr = loadDataSet("testSetRBF2.txt")
    errorCount = 0
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m,n = shape(dataMat)

    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], \
                                 ("rbf", _k1))
        predict = kernelEval.T * multiply(labelSV, \
                                          alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("The test error rate is: %f" % \
            (float(errorCount /m)))

def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(_dirName):
    hwLabels = []
    trainingFileList = listdir(_dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector("%s/%s" % \
                            (_dirName, fileNameStr))
    return trainingMat, hwLabels 

def testDigits(_kTup = ("rbf", 10)):
    dataArr, labelArr = loadImages("trainingDigits")
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, \
                     10000, _kTup)
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d support vectors" % shape(sVs)[0])
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], _kTup)
        predict = kernelEval.T * multiply(labelSV, \
                                          alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("The training error rate is: %f" % \
          (float(errorCount) / m))
    dataArr, labelArr = loadImages("testDigits")
    errorcount = 0
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], _kTup)
        predict = kernelEval.T * multiply(labelSV, \
                                          alphas[svInd]) + b
        if (sign(predict) != sign(labelArr[i])):
            errorCount += 1
    print("The test error rate is: %f" % \
          (float(errorCount) / m))

def main():
    #e.g. 1 unoptimized
    #dataArr, labelArr = loadDataSet("testSet.txt")
    #b, alphas =  smoSimple(dataArr, labelArr, .6, .001, 40)
    #print(b)
    #print(alphas)
    #print(alphas[alphas > 0])
    #print(shape(alphas[alphas > 0]))
    #for x in range(100):
    #    if alphas[x] > 0.0:
    #        print(dataArr[x], labelArr[x])

    #e.g. 2 optimized
    #dataArr, labelArr = loadDataSet("testSet.txt")
    #b, alphas = smoP(dataArr, labelArr, .6, .001, 40)
    #print("b:", b)
    #ws = calcWs(alphas, dataArr, labelArr)
    #print("weights:", ws)
    #dataMat = mat(dataArr)
    
    # classification.
    #for x in range(5):
    #    print("Prediction:", "1" if dataMat[x] * \
    #          mat(ws) + b > 0 else "-1")
    #    print("Actual:", labelArr[x])

    #e.g. 3 Radial Bias Function Kernel
    #    testRbf()

    #e.g. 4
    testDigits(("rbf", 20))
main()
