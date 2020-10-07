import matplotlib.pyplot as plt
from numpy import *


#加载数据集,z=w0*x0+w1*x1+w2*x2+.....=(w0,w1,w2,...)*(x0,x1,x2,...)T，其中x1设为1.0
def loadDataSet():#数据集的前两个值分别为x1和x2，第三个值是数据对应的类别标签，x0设为1.0
    dataMat = [];labelMat = []#数据集和标签集
    fr = open('testSet.txt')#读取数据集文件
    for line in fr.readlines():
        lineArr = line.strip().split()#默认按空格分割
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat
'''
def sigmoid(inX):#sigmoid函数的定义
    return 1.0/(1+exp(-inX))
'''
def sigmoid(inX):#sigmoid函数的定义
    from numpy import exp
    if inX>=0:
        return 1.0/(1+exp(-inX))
    else:
        return exp(inX)/(1+exp(inX))

#梯度上升算法，用来计算初最佳回归系数
#第一个参数是2维数组，每列代表每个不同特征，每行代表每个训练样本
#第二个参数是类别标签，1*100的行向量，为了方便计算，进行转置
def gradAscent(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn)#获得输入数据并将其转换为numpy矩阵数据类型,100*3
    labelMat = mat(classLabels).transpose()#获得标签并将其转换为numpy矩阵数据类型,1*100行向量,进行转置，变成100*1的列向量
    m,n = shape(dataMatrix)#返回矩阵的维数
    alpha = 0.001#步长，向函数增长最快的方向的移动量，即学习率
    maxCycles = 500#迭代次数
    weights = ones((n,1))#生成n行1列的元素为1的矩阵赋值给weights,即回归系数初始化为1
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)#矩阵相乘，100*1的列向量
        error = (labelMat - h)#计算真实类别与预测类别的差值，h是一个列向量，列向量的元素个数等于样本数，即100
        weights = weights+alpha*dataMatrix.transpose()*error#*dataMatrix.transpose()*error是梯度，按照该差值的方向调整回归系数
    return weights

#>>> import logRegres
#>>> dataArr,labelMat = logRegres.loadDataSet()
#>>> logRegres.gradAscent(dataArr,labelMat)

#画点,画线
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat = loadDataSet()#加载数据
    dataArr = array(dataMat)
    n = shape(dataArr)[0]#100行数据
    xcord1 = [];ycord1 = []#存储类别为1的点
    xcord2 = [];ycord2 = []#存储类别为0的点
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])

    #画点
    fig = plt.figure()
    ax = fig.add_subplot(111)#准备画布
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')#类别为1的画红点
    ax.scatter(xcord2,ycord2,s=30,c='green')#类别为0的画绿点
    #画线
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0]*1.0-weights[1]*x)/weights[2]#最佳拟合直线

    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()#显示

'''
>>> import logRegres
>>> dataArr,labelMat = logRegres.loadDataSet()
>>> weights = logRegres.gradAscent(dataArr,labelMat)
>>> weights
matrix([[ 4.12414349],
        [ 0.48007329],
        [-0.6168482 ]])
>>> logRegres.plotBestFit(weights.getA())
'''
    

def GetResult():
    dataMat,labelMat = loadDataSet()
    weights = gradAscent(dataMat,labelMat)
    print(weights)
    plotBestFit(weights.getA())
#>>> import logRegres
#>>> logRegres.GetResult()

#随机梯度上升算法得到的h和error都是数值，而且没有矩阵转换过程，全部都是numpy数组
def stocGradAscent0(dataMatrix,classLabels):
    m,n =  shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h=sigmoid(sum(dataMatrix[i]*weights))#对于单个点求sigmoid函数值
        error = classLabels[i] - h
        weights = weights+alpha*error*dataMatrix[i]

    return weights

#>>> from numpy import *
#>>> import logRegres
#>>> dataArr,labelMat = logRegres.loadDataSet()
#>>> weights = logRegres.stocGradAscent0(array(dataArr),labelMat)
#>>> logRegres.plotBestFit(weights)

#改进随机梯度上升算法
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01#alpha在每次迭代时都会调整
            randIndex = int(random.uniform(0,len(dataIndex)))#随机选一组数据
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights+alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])#删除已经训练过的数据
    return weights
            
'''
注：range返回的时range对象，不反悔数组对象
trainingSet = range(50) 改为 trainingSet = list(range(50))

>>> from numpy import *
>>> import logRegres
>>> dataArr,labelMat = logRegres.loadDataSet()
>>> weights = logRegres.stocGradAscent1(array(dataArr),labelMat)
>>> logRegres.plotBestFit(weights)
>>> weights = logRegres.stocGradAscent1(array(dataArr),labelMat,500)
>>> logRegres.plotBestFit(weights)
'''

def classifyVector(inX,weights):#分类算法，sigmoid函数大于0.5，分类结果为1，小于0.5为0
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():#测试算法
    frTrain = open('horseColicTraining.txt')#打开训练数据集
    frTest = open('horseColicTest.txt')#打开测试数据集
    trainingSet = [];trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')#以制表符分隔训练数据
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))#前20列为特征数据
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))#20个特征，下标21表示分类标签
    trainWeights = stocGradAscent1(array(trainingSet),trainingLabels,500)#训练数据集，得到相应的weight
    errorCount = 0;numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0#每次进行一组数据的测试，统计测试次数
        currLine = line.strip().split('\t')#以制表符分隔测试数据
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights))!= int(currLine[21]):
            errorCount += 1#统计错误次数
    errorRate = (float(errorCount)/numTestVec)
    print('the error rate of this test is:%f '%errorRate)
    return errorRate

def multiTest():#测试10次，求平均值
    numTests = 10;errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print('after %d iteration the average error rate is :%f'%(numTests,errorSum/float(numTests)))

'''
>>> import logRegres
>>> logRegres.colicTest()
the error rate of this test is:0.373134 
0.373134328358209
>>> logRegres.multiTest()
'''
    
        



    
    

