from numpy import *

def loadSimpleData():
    datMat = matrix([
        [1.,2.1],
        [2.,1.1],
        [1.3,1.],
        [1.,1.],
        [2.,1.]
        ])

    classLabels = [1.0,1.0,-1.0,-1.0,1.0]

    return datMat,classLabels

'''
>>> import adaboost
>>> datMat,classLabels = adaboost.loadSimpleData()

'''

#首先是在最小值的基础上一步步增加步长得到阈值，然后求出将哪一类设为=1,的误差更小，得到误差最小的阈值，和最好的分类标签

#单层决策树的阈值过滤函数
#shape(datMat)，矩阵的维度；zeros((m,1))，创建m*1的矩阵，数值设为0，ones((m,1)),m*1的矩阵，数值设为1
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = ones((shape(dataMatrix)[0],1))#对数据集每一列的各个特征进行阈值过滤
    if threshIneq == 'lt':#将小于某一阈值的特征归类为-1
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:#将大于某一阈值的特征归类为-1
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0

    return retArray


def buildStump(dataArr,classLabels,D):#D为数据权值分布
    dataMatrix = mat(dataArr);labelMat = mat(classLabels).T#将数据集和标签列表转为矩阵形式
    m,n = shape(dataMatrix)
    numSteps = 10.0;bestStump = {};bestClassEst = mat(zeros((m,1)))#存储最好的分类标签，m行1列，数值为0的矩阵
    #步长或区间总数，最优决策树信息，最优单层决策树预测结果
    minError = inf#最小错误率初始化为正无穷
    for i in range(n):#遍历每一列的特征值
        rangeMin = dataMatrix[:,i].min()#找出列中特征值的最小值
        rangeMax = dataMatrix[:,i].max()#最大值
        stepSize = (rangeMax - rangeMin)/numSteps#求取步长大小或者区间间隔，控制阈值每次增加多少
        for j in range(-1,int(numSteps)+1):#遍历各个步长区间
            for inequal in ['lt','gt']:#两种阈值过滤模式
                threshVal = (rangeMin+float(j)*stepSize)#阈值计算公式：最小值+j(-1<=j<=numsteps+1)*步长
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#选定阈值后，调用阈值过滤函数分类预测
                #print(predictedVals)
                errArr = mat(ones((m,1)))#初始化错误向量
                errArr[predictedVals == labelMat] = 0#将错误向量中分类正确项置0
                #print(errArr)

                weightedError = D.T *errArr#计算加权错误率
                #print('split:dim %d,thresh %.2f,thresh ineqal: %s,the weighted error is %.3f'%(i,threshVal,inequal,weightedError))
                if weightedError < minError:#如果当前错误率小于当前最小错误率，将当前错误率作为最小错误率
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClassEst#返回最佳单层决策树相关信息的字典，最小错误率，决策树预测输出结果
                

'''
>>> import adaboost
>>> datMat,classLabels = adaboost.loadSimpleData()
>>> from numpy import *
>>> D = mat(ones((5,1))/5)
>>> D
matrix([[0.2],
        [0.2],
        [0.2],
        [0.2],
        [0.2]])
>>> adaboost.buildStump(datMat,classLabels,D)
'''

#基于单层决策树的adaboost训练过程
#（数据矩阵，标签向量，迭代次数）

def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []#弱分类器相关信息列表
    m = shape(dataArr)[0]#数据集的行数
    D = mat(ones((m,1))/m)#初始化权重向量的每一项值相等
    aggClassEst = mat(zeros((m,1)))#累计估计值向量
    for i in range(numIt):#迭代循环次数
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#根据当前数据集、标签以及权重建立最佳单层决策树
        print('D',D.T)#打印权重向量
        alpha = float(0.5*log((1.0-error)/(max(error,1e-16))))#求单层决策树的系数alpha
        bestStump['alpha'] = alpha#存储决策树的系数alpha到字典
        weakClassArr.append(bestStump)#将该决策树存入列表
        #print('classEst',classEst.T)#打印决策树的预测结果，即每个弱分类器的预测结果
        #预测正确为exp(-alpha)，预测错误为exp(alpha)，增加分类错误的样本权重，减少分类正确的数据点权重
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))
        D = D/D.sum()#更新权值向量
        aggClassEst += alpha*classEst#累加当前单层决策树的加权预测值
        print('aggClassEst',aggClassEst.T)
        #求出分类错误的样本个数
        aggErrors = multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))
        #print(sign(aggClassEst))

        #print(aggErrors)

        errorRate = aggErrors.sum()/m#计算错误率
        print('total error:',errorRate,'\n')
        if errorRate == 0.0:break#错误率为0.0退出循环
    #return weakClassArr#返回弱分类器的组合列表
    return weakClassArr,aggClassEst#修改代码，可以直接看出弱分类器组合后的分类预测

'''
>>> import adaboost
>>> datMat,classLabels = adaboost.loadSimpleData()
>>> classifierArray,aggClassEst = adaboost.adaBoostTrainDS(datMat,classLabels,9)
>>> classifierArray

'''

#测试adaBoost,adaBoost分类函数，（测试数据点，构建好的最终分类器）

def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)#构建数据矩阵
    m = shape(dataMatrix)[0]#获取矩阵行数
    aggClassEst = mat(zeros((m,1)))#初始化分类器
#   for i in range(len(classifierArr[0])):#遍历分类器列表中的每一个弱分类器
    for i in range(len(classifierArr)):#遍历分类器列表中的每一个弱分类器

        #对每一个弱分类器对测试数据进行预测分类
        #classEst = stumpClassify(dataMatrix,classifierArr[0][i]['dim'],#python3中，加个[0]
                                 #classifierArr[0][i]['thresh'],
                                 #classifierArr[0][i]['ineq'])
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        #print(classEst)
        #aggClassEst += classifierArr[0][i]['alpha']*classEst#对各个分类器的预测结果进行加权累加
        aggClassEst += classifierArr[i]['alpha']*classEst#对各个分类器的预测结果进行加权累加

        print('aggClassEst',aggClassEst)
    return sign(aggClassEst)#通过sign函数根据结果大于或小于0预测+1或-1    
    
'''
>>> import adaboost
>>> datArr,labelArr = adaboost.loadSimpleData()
>>> #classifierArr = adaboost.adaBoostTrainDS(datArr,labelArr,30)
>>> classifierArr,aggClassEst = adaboost.adaBoostTrainDS(datArr,labelArr,30)

D [[0.2 0.2 0.2 0.2 0.2]]
classEst [[-1.  1. -1. -1.  1.]]
aggClassEst [[-0.69314718  0.69314718 -0.69314718 -0.69314718  0.69314718]]
total error: 0.2 

D [[0.5   0.125 0.125 0.125 0.125]]
classEst [[ 1.  1. -1. -1. -1.]]
aggClassEst [[ 0.27980789  1.66610226 -1.66610226 -1.66610226 -0.27980789]]
total error: 0.2 

D [[0.28571429 0.07142857 0.07142857 0.07142857 0.5       ]]
classEst [[1. 1. 1. 1. 1.]]
aggClassEst [[ 1.17568763  2.56198199 -0.77022252 -0.77022252  0.61607184]]
total error: 0.0 

>>> adaboost.adaClassify([0,0],classifierArr)
aggClassEst [[-0.69314718]]
aggClassEst [[-1.66610226]]
aggClassEst [[-2.56198199]]
matrix([[-1.]])
>>> adaboost.adaClassify([5,5],classifierArr)
aggClassEst [[0.69314718]]
aggClassEst [[1.66610226]]
aggClassEst [[2.56198199]]
matrix([[1.]])
>>> classifierArr
[{'dim': 0, 'thresh': 1.3, 'ineq': 'lt', 'alpha': 0.6931471805599453}, {'dim': 1, 'thresh': 1.0, 'ineq': 'lt', 'alpha': 0.9729550745276565}, {'dim': 0, 'thresh': 0.9, 'ineq': 'lt', 'alpha': 0.8958797346140273}]
'''

#测试马疝病实例

#加载数据集

def loadDataSet(fileName):
    #获取特征数目，包括最后一列标签，readline(读取文件的一行)，readlines(读取整个文件所有行)
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = [];labelMat = []#创建数据集矩阵，标签向量
    fr = open(fileName)
    for line in fr.readlines():#遍历文本的每一行
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)#添加数据矩阵
        labelMat.append(float(curLine[-1]))#添加标签向量
    return dataMat,labelMat

'''
>>> import adaboost
>>> datArr,labelArr = adaboost.loadDataSet('horseColicTraining2.txt')
>>> #classifierArray = adaboost.adaBoostTrainDS(datArr,labelArr,10)
>>> classifierArray,aggClassEst = adaboost.adaBoostTrainDS(datArr,labelArr,10)

>>> testArr,testLabelArr = adaboost.loadDataSet('horseColicTest2.txt')
>>> prediction10 = adaboost.adaClassify(testArr,classifierArray)
>>> prediction10

>>> from numpy import *
>>> errArr = mat(ones((67,1)))
>>> errArr[prediction10!=mat(testLabelArr).T].sum()

'''
#测试函数

def classify():
    datArr,labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArray,aggClassEst = adaBoostTrainDS(datArr,labelArr,10)
    print(classifierArray)
    
    testArr,testLabelArr = loadDataSet('horseColicTest2.txt')
    prediction10 = adaClassify(testArr,classifierArray)
    num = shape(mat(testLabelArr))[1]
    errArr = mat(ones((67,1)))
    error = errArr[prediction10!=mat(testLabelArr).T].sum()
    print(num,'hhhhhh')
    errorRate=float(error)/float((num))
    print("the errorRate is: %.2f",errorRate)

'''
>>> import adaboost
>>> adaboost.classify()
'''
#ROC曲线的绘制以及AUC（曲线下面积）计算函数
def plotROC(predStrengths,classLabels):#输入（预测强度，分类结果），输出ROC图
    import matplotlib.pyplot as plt
    cur = (1.0,1.0)#当前绘制节点,cur[0]、cur[1]分别代表x轴，y轴
    ySum = 0.0#AUC统计
    numPosClas = sum(array(classLabels)==1.0)#实际为+1的分类数
    #print(array(classLabels)==1.0)
    #print(numPosClas)
    yStep = 1/float(numPosClas)#x轴移动步长,+1
    xStep = 1/float(len(classLabels)-numPosClas)#y轴移动步长,-1
    sortedIndicies = predStrengths.argsort()#预测强度排序（下标排序）
    fig = plt.figure()#设置画布
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:#以预测强度递减的次序绘制ROC统计图像
        if classLabels[index] == 1.0:#预测为1的结果，从点（1，1）开始画，慢慢减少
            delX = 0;delY = yStep;
            #print(delY,'ystep')
        else:
            delX = xStep;delY = 0;
            print(delX,'xstep')
            ySum += cur[1]
            #print(ySum,'ySum')
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')#画00到11的直线，[x1,x2],[y1,y2]
    plt.xlabel('False Positive Rate');plt.ylabel('True Positive Rate')
    plt.title('ROC curvve for Adaboost Horse Colic Detection System')
    ax.axis([0,1,0,1])#axis([xmin xmax ymin ymax])，设置xy坐标轴范围 
    plt.show()
    print('the Area Under the Curve is:',ySum*xStep)


'''
 plot(x,y,'--gs','LineWidth',2,'MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor',[0.5,0.5,0.5]) 
'''


'''
>>> import adaboost
>>> datArr,labelArr = adaboost.loadDataSet('horseColicTraining2.txt')
>>> classifierArray,aggClassEst = adaboost.adaBoostTrainDS(datArr,labelArr,10)
>>> adaboost.plotROC(aggClassEst.T,labelArr)

'''
    




