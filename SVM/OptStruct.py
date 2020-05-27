# Machine Learning in Action by Peter Harrington
# Manning Publication
# Chapter 6
# Sequential minimal optimization

from numpy import *
#from svm import kernelTrans
#import svm as svm

class OptStruct:
    def __init__(self, _dataMatIn, _classLabels, \
                 _C, _toler, _kTup):
        self.X = _dataMatIn
        self.labelMat = _classLabels
        self.C = _C
        self.tol = _toler
        self.m = shape(_dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        #(valid, value), valid -> calculated
        self.eCache = mat(zeros((self.m, 2)))
        self.k = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.k[:, i] = self.kernelTrans(self.X, \
                            self.X[i, :], _kTup)

    def kernelTrans(_self, _X, _A, _kTup):
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
