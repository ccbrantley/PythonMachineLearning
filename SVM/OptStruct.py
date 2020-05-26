# Machine Learning in Action by Peter Harrington
# Manning Publication
# Chapter 6
# Sequential minimal optimization

from numpy import *

class OptStruct:
    def __init__(self, _dataMatIn, _classLabels, _C, _toler):
        self.X = _dataMatIn
        self.labelMat = _classLabels
        self.C = _C
        self.tol = _toler
        self.m = shape(_dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        #(valid, value), valid -> calculated
        self.eCache = mat(zeros((self.m, 2)))
