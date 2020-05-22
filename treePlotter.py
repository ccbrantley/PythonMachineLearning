# Machine Learning in Action by Peter Harrington.
# Manning publications.
# Chapter 3.

import matplotlib.pyplot as plt

decisionNode = dict(boxstyle = "sawtooth", fc = ".8")
leafNode = dict(boxstyle = "round4", fc = ".8")
arrowArgs = dict(arrowstyle = "<-")

def plotNode(_nodeTxt, _centerPt, _parentPt, _nodeType):
    createPlot.ax1.annotate(_nodeTxt, xy = _parentPt, \
                            xycoords = 'axes fraction', \
                            xytext = _centerPt, \
                            textcoords = "axes fraction", \
                            va = "center", ha = "center", \
                            bbox = _nodeType, \
                            arrowprops = arrowArgs)

def plotMidText(_cntrPt, _parentPt, _txtString):
    xMid = (_parentPt[0] - _cntrPt[0]) / 2.0 + _cntrPt[0]
    yMid = (_parentPt[1] - _cntrPt[1]) / 2.0 + _cntrPt[1]
    createPlot.ax1.text(xMid, yMid, _txtString)

def plotTree(_myTree, _parentPt, _nodeTxt):
    numLeafs = getNumLeafs(_myTree)
    getTreeDepth(_myTree)
    firstStr = list(_myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 \
              / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, _parentPt, _nodeTxt)
    plotNode(firstStr, cntrPt, _parentPt, decisionNode)
    secondDict = _myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 \
                            / plotTree.totalW
            plotNode(secondDict[key], \
                     (plotTree.xOff, plotTree.yOff), \
                     cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), \
                        cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

def createPlot(_inTree):
    fig = plt.figure(1, facecolor = "white")
    fig.clf()
    axprops = dict(xticks = [], yticks = [])
    createPlot.ax1 = plt.subplot(111, frameon = False, \
                                 **axprops)
    plotTree.totalW = float(getNumLeafs(_inTree))
    plotTree.totalD = float(getTreeDepth(_inTree))
    plotTree.xOff = -.5 / plotTree.totalW
    plotTree.yOff = 1.0;
    plotTree(_inTree, (.5, 1.0), "")
    plt.show()

def getNumLeafs(_myTree):
    numLeafs = 0
    firstStr = list(_myTree.keys())[0]
    secondDict = _myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(_myTree):
    maxDepth = 0
    firstStr = list(_myTree.keys())[0]
    secondDict = _myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

def retrieveTree(i):
    listOfTrees = [{"no surfacing": {0: "no", 1: {"flippers": \
                   {0: "no", 1: "yes"}}}},
                   {"no surfacing": {0: "no", 1: {"flippers": \
                   {0: {"head": {0: "no", 1: "yes"}}, 1: "no"}}
                   }}
                  ]
    return listOfTrees[i]
