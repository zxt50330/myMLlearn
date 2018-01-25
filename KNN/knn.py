from numpy import *
from numpy import array
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
#
# group, labels = createDataSet()
# print(classify0([0,0], group, labels, 3))


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)         # get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        # prepare matrix to return
    classLabelVector = []                       # prepare labels return
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

# datingDataMat, datingLabels = file2matrix('./datingTestSet.txt')
# print(datingDataMat, datingLabels)


def autoNorm(dataSet):
    """
    归一化
    newValue = (oldValue-min)/(max-min)
    :type dataSet: array
    :return:
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]  # 行数
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals
# print(autoNorm(datingDataMat))


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('./datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)  #  取1/10
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 7)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)

# datingClassTest()


def classifyPerson():
    resultList = [1,2,3]
    percentTats = float(input("games"))
    ffMiles = float(input("frequent"))
    iceCream = float(input("ice"))
    datingDataMat, datingLabels = file2matrix('./datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 7)
    print(resultList[classifierResult-1])


import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111) # 画布位置 将画布分割成1行1列，图像画在从左到右从上到下的第1块
# ax.scatter(datingDataMat[:,0], datingDataMat[:,1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
# plt.show()


