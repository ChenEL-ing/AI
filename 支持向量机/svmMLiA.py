from numpy import *
def loadDataSet(fileName):#数据集前两列时特征数据，后一列是标签数据
    dataMat = [];labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split()#同逻辑斯谛回归数据提取方式
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat
#两个辅助函数，一个用于在某个区间范围内随机选择一个整数，另一个是用于数值太大时对其进行调整

def selectJrand(i,m):#i是第一个alpha的下标，m是所有alpha的数目
    j = i
    while(j==i):#在样本集中采取随机选择的方法选取第二个不等于第一个alphai的优化向量alphaj
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):#调整大于H或小于L的alpha值,控制aj的值在（L,H）之间
    if(aj>H):
        aj = H
    if(L>aj):
        aj = L
    return aj

'''
>>> svmMLiA.selectJrand(1,100)
60
>>> svmMLiA.selectJrand(66,100)
87

>>> svmMLiA.clipAlpha(50,60,40)
50
>>> svmMLiA.clipAlpha(66,60,40)
60
>>> svmMLiA.clipAlpha(36,60,40)
40
'''


'''
>>> import svmMLiA
>>> dataArr,labelArr = svmMLiA.loadDataSet('svmtestSet.txt')
>>> labelArr
[-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0]
>>> dataArr

'''

#@dataMat:数据列表
#@classLabels:标签列表
#@C:权衡因子（增加松弛因子而在目标优化函数中引入惩罚项）
#@toler:容错率
#@maxIter:最大迭代次数
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix = mat(dataMatIn)#把列表形式转为矩阵或向量形式
    labelMat = mat(classLabels).transpose()
    b = 0#初始截距为0
    m,n = shape(dataMatrix)#获得矩阵行列
    alphas = mat(zeros((m,1)))#新建一个m*1的0向量
    iter = 0#记录迭代次数
    while(iter<maxIter):
        alphaPairsChanged = 0#改变的alpha对数
        for i in range(m):#遍历样本集中样本
            #计算支持向量机算法的预测值
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b
            Ei = fXi - float(labelMat[i])#计算预测值与实际值的误差
            #如果不满足KKT条件，即labelMat[i]*fXi<1(labelMat[i]*fXi-1<-toler)
            #and alpha<C 或者labelMat[i]*fXi>1(labelMat[i]*fXi-1>toler)and alpha>0
            if((labelMat[i]*Ei < toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
              j = selectJrand(i,m)#随机选择第二个变量alphaj
              #计算第二个变量对应数据的预测值
              fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b
              Ej = fXj - float(labelMat[j])#计算预测值与实际值的误差
              alphaIold = alphas[i].copy()
              alphaJold = alphas[j].copy()#记录alphai和alphaj的原始值，便于后续比较
              if(labelMat[i]!=labelMat[j]):#如果两个alpah对应样本的标签不相同
                  L = max(0,alphas[j]-alphas[i])
                  H = min(C,C+alphas[j]-alphas[i])
              else:#如果两个alpah对应样本的标签不相同，求出相应的上下边界
                  L = max(0,alphas[j]+alphas[i]-C)
                  H = min(C,alphas[j]+alphas[i])

              if L==H:print('L=H');continue
              #根据公式计算未经剪辑的alphaj

              eta = 2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]*dataMatrix[j,:].T
              if eta>=0:print('eta>=0');continue
              alphas[j] -= labelMat[j]*(Ei-Ej)/eta
              alphas[j] = clipAlpha(alphas[j],H,L)#进行剪辑

              if(abs(alphas[j]-alphaJold)<0.00001):print('j not moving enough');continue
              #否则，计算相应的alphai的值
              alphas[i] += labelMat[i]*labelMat[j]*(alphaJold-alphas[j])

              #再分别计算两个alpha情况下对应的b值

              b1 = b-Ei-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
              b2 = b-Ej-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T

              if(0<alphas[i]) and (C>alphas[i]):b = b1
              elif (0<alphas[j]) and (C>alphas[j]):b = b2
              else:b = (b1+b2)/2.0

              alphaPairsChanged += 1
              print('iter :%d i:%d,pairs changed %d'%(iter,i,alphaPairsChanged))
        #判断是否有改变的alpha对，没有就进行下一次迭代
        if(alphaPairsChanged == 0):iter += 1
        #否则，迭代次数置0，继续循环
        else:iter = 0
        print('iteration number:%d'%iter)
    #返回最后的b和alpha向量
    return b,alphas

''''
>>> import svmMLiA
>>> dataArr,labelArr = svmMLiA.loadDataSet('svmtestSet.txt')
>>> b,alphas = svmMLiA.smoSimple(dataArr,labelArr,0.6,0.001,40)
>>> b
matrix([[-2.69242065]])
>>> alphas[alphas>0]
matrix([[0.13789329, 0.02835488, 0.10953677]])

'''
'''
>>> from numpy import *
>>> shape(alphas[alphas>0])
(1, 3)
>>> for i in range(40):
	if alphas[i]>0.0:print(dataArr[i],labelArr[i])
'''

#启发式SMO算法的支持函数
#新建一个类的数据结构，保存当前重要的值
class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2)))
        
#格式化计算五误差的函数，方便多次调用
def calcEk(oS,k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T))+oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

'''
>>> from numpy import *
>>> a = mat(zeros((4,2)))
>>> a
matrix([[0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.]])

>>> a[0] = [1,1]
>>> a[1] = [1,2]
>>> a[2] = [1,3]
>>> a
matrix([[1., 1.],
        [1., 2.],
        [1., 3.],
        [0., 0.]])
>>> b = nonzero(a[:,0].A)[0]
>>> b
array([0, 1, 2], dtype=int64)
>>> for i in b:
	print(i)

	
0
1
2

获取不为0的下标i，即获取Ei不为0的alpha
'''

#修改选择第二个变量alphaj的方法，返回第二个变量和误差
def selectJ(i,oS,Ei):
    maxK = -1;maxDeltaE = 0;Ej  = 0
    oS.eCache[i] = [1,Ei]#误差矩阵的第一列给出的是eCache是否是有效位，而第二列给出的是实际的E值
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]#获取缓存中Ei不为0的样本对应的alpha列表
    #在误差不为0的列表中找出使abs(Ei-Ej)最大的alphaj
    if(len(validEcacheList)>0):
        for k in validEcacheList:
            if k==i:continue  #j=i时，重新选择j
            Ek = calcEk(oS,k)#j以及选好，计算Ei
            deltaE = abs(Ei-Ek)
            if(deltaE>maxDeltaE):#选取Ei-Ej差最大的
                maxK = k;maxDeltaE = deltaE;Ej = Ek
        return maxK,Ej
    else:#否则，就从样本集中随机选取alphaj
        j = selectJrand(i,oS,m)
        Ej = calcEk(oS,j)
    return j,Ej

#更新误差矩阵
def updateEk(oS,k):
 Ek = calcEk(oS,k)
 oS.eCache[k] = [1,Ek]

#内循环寻找决策边界的优化，如果alphai，alphaj对改变返回1，否则返回0
def innerL(i,oS):
    Ei = calcEk(oS,i)#计算误差
    if(((oS.labelMat[i]*Ei<-oS.tol)and(oS.alphas[i]<oS.C))or((oS.labelMat[i]*Ei>oS.tol)and(oS.alphas[i]>0))):
        j,Ej = selectJ(i,oS,Ei)
        alphaIold = oS.alphas[i].copy();alphaJold = oS.alphas[j].copy()
        if(oS.labelMat[i]!=oS.labelMat[j]):
            L = max(0,oS.alphas[j]-oS.alphas[i])
            H = min(oS.C,oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L = max(0,oS.alphas[j]+oS.alphas[i]-oS.C)
            H = min(oS.C,oS.alphas[j]+oS.alphas[i])
        if L==H:print('L=H');return 0
        #计算两个alpha的值
        eta = 2.0*oS.X[i,:]*oS.X[j,:].T-oS.X[i,:]*oS.X[i,:].T-oS.X[j,:]*oS.X[j,:].T
        if eta>=0:print('eta>=0');return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS,j)

        if(abs(oS.alphas[j]-alphaJold)<0.00001):
            print('j not moving enough');return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold-oS.alphas[j])
        updateEk(oS,i)
        #在这两个alpha值情况下，计算对应的b值

        b1 = oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T-\
             oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T-\
             oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T


        if(oS.alphas[i]>0) and (oS.alphas[i]<oS.C):oS.b = b1
        elif(oS.alphas[j]>0) and (oS.alphas[j]<oS.C):oS.b = b2
        else:oS.b = (b1+b2)/2.0

        #如果有alpha对更新,返回1，否则返回0
        return 1
    else:return 0


#SMO外循环代码，交替在全部数据集上和非边界上数据集上遍历，用entireSet控制True、False交替

def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler)
    iter = 0
    entireSet = True;alphaPairsChanged = 0
    #选取第一个变量alpha的三种情况，从间隔边界上选取或整个数据集

    while(iter<maxIter) and ((alphaPairsChanged>0)or(entireSet)):
        alphaPairsChanged = 0#没有alpha更新对
        if entireSet:
            for i in range(oS.m):#在全部数据集上遍历
                alphaPairsChanged += innerL(i,oS)
                print('fullset,iter:%d i:%d,pairschanged %d'%(iter,i,alphaPairsChanged))
            iter += 1
        else:#在非边界上遍历
            nonBoundIs = nonzero((oS.alphas.A)>0*(oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged+=innerL(i,oS)
                print('non-bound,iter:%d i:%d,pairschanged %d'%(iter,i,alphaPairsChanged))

            iter += 1
        if entireSet:entireSet = False
        #如果本次循环没有改变的alpha对，将entireSet设置为true
        elif(alphaPairsChanged==0):entireSet = True
        print('iteration number:%d'%iter)
    return oS.b,oS.alphas
    
'''
>>> import svmMLiA
>>> dataArr,labelArr = svmMLiA.loadDataSet('svmtestSet.txt')
>>> b,alphas = svmMLiA.smoP(dataArr,labelArr,0.6,0.001,40)
'''
#计算w
def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr);labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w
    
'''
>>> import svmMLiA
>>> dataArr,labelArr = svmMLiA.loadDataSet('svmtestSet.txt')
>>> b,alphas = svmMLiA.smoP(dataArr,labelArr,0.6,0.001,40)
>>> ws = svmMLiA.calcWs(alphas,dataArr,labelArr)
>>> ws
array([[ 0.44996986],
       [-0.11314321]])
>>> from numpy import *
>>> datMat = mat(dataArr)
>>> datMat[0]*mat(ws)+b
matrix([[-1.14248703]])
>>> labelArr[0]
-1.0
>>> from numpy import *
>>> datMat = mat(dataArr)
>>> datMat[0]*mat(ws)+b
matrix([[-1.14248703]])
>>> labelArr[0]
-1.0

'''

#核函数转换

def kernelTrans(X,A,kTup):
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if(kTup[0]=='lin'):#如果核函数类型为'lin'，线性核
        K=X*A.T
    elif kTup[0] == 'rbf':#如果核函数类型为'rbf':径向基函数
        for j in range(m):#将每个样本向量利用核函数转为高维空间
            deltaRow = X[j,:]-A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2))
    else:raise NameError('Houston we have a problem--That Kernel is not recognised')

    return K

class optStruct1:
    def __init__(self,dataMatIn,classLabels,C,toler,kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2)))
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            for i in range(self.m):
                self.K[:,i] = kernelTrans(self.X,self.X[i,:],kTup)

def smoP1(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
    oS = optStruct1(mat(dataMatIn),mat(classLabels).transpose(),C,toler,kTup)
    iter = 0
    entireSet = True;alphaPairsChanged = 0
    #选取第一个变量alpha的三种情况，从间隔边界上选取或整个数据集

    while(iter<maxIter) and ((alphaPairsChanged>0)or(entireSet)):
        alphaPairsChanged = 0#没有alpha更新对
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL1(i,oS)
                print('fullset,iter:%d i:%d,pairs changed %d'%(iter,i,alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A)>0*(oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged+=innerL1(i,oS)
                print('non-bound,iter:%d i:%d,pairs changed %d'%(iter,i,alphaPairsChanged))

            iter += 1
        if entireSet:entireSet = False
        #如果本次循环没有改变的alpha对，将entireSet设置为true
        elif(alphaPairsChanged==0):entireSet = True
        print('iteration number:%d'%iter)
    return oS.b,oS.alphas

def calcEk1(oS,k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k]+oS.b)
    Ek = fXk-float(oS.labelMat[k])
    return Ek
    
def innerL1(i,oS):
    Ei = calcEk1(oS,i)#计算误差
    if(((oS.labelMat[i]*Ei<-oS.tol)and(oS.alphas[i]<oS.C))or((oS.labelMat[i]*Ei>oS.tol)and(oS.alphas[i]>0))):
        j,Ej = selectJ(i,oS,Ei)
        alphaIold = oS.alphas[i].copy();alphaJold = oS.alphas[j].copy()
        #计算上下界
        if(oS.labelMat[i]!=oS.labelMat[j]):
            L = max(0,oS.alphas[j]-oS.alphas[i])
            H = min(oS.C,oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L = max(0,oS.alphas[j]+oS.alphas[i]-oS.C)
            H = min(oS.C,oS.alphas[j]+oS.alphas[i])
        if L==H:print('L==H');return 0
        #计算两个alpha的值
        eta = 2.0*oS.K[i,j]-oS.K[i,i]-oS.K[j,j]
        if eta>=0:print('eta>=0');return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS,j)
        if(abs(oS.alphas[j]-alphaJold)<0.0001):
            print('j not moving enough');return 0;
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold-oS.alphas[j])
        updateEk(oS,i)

        b1 = oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i]-\
             oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]-\
             oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if(oS.alphas[i]>0) and (oS.alphas[i]<oS.C):oS.b = b1
        elif(oS.alphas[j]>0) and (oS.alphas[j]<oS.C):oS.b = b2
        else:oS.b = (b1+b2)/2.0
        return 1
    else:return 0

#测试核函数，用户指定到达率
def testRbf(k1=1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoP1(dataArr,labelArr,200,0.0001,10000,('rbf',k1))
    dataMat = mat(dataArr);labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A>0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print('there are %d Support Vectors'%shape(sVs)[0])
    m,n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict = kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if sign(predict)!= sign(labelArr[i]):errorCount+=1
    print('the training error rate is:%f'%(float(errorCount)/m))

    #第二个测试集
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    dataMat = mat(dataArr);labelMat = mat(labelArr).transpose()
    errorCount = 0
    m,n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict = kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if sign(predict)!=sign(labelArr[i]):errorCount+=1
    print('the training error rate is :%f'%(float(errorCount)/m))

'''
>>> import svmMLiA
>>> svmMLiA.testRbf()
'''

#识别数字

#图片向量化，对每个32*32的数字向量化为1*1024
def img2vector(filename):
    returnVect = zeros((1,1024))#numpy矩阵，1*1024
    fr = open(filename)#使用open函数打开一个文本文件
    for i in range(32):#循环读取文件内容
        lineStr = fr.readline()#读取一行，返回字符串
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])#循环放入1*1024矩阵中
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:hwLabels.append(-1)
        else:hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s'%(dirName,fileNameStr))
    return trainingMat,hwLabels


#利用svm测试数字

def testDigits(kTup=('rbf',10)):
    #训练集
    dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = smoP1(dataArr,labelArr,200,0.0001,10000,kTup)
    dataMat = mat(dataArr);labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A>0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print('there are %d Support Vectors'%(shape(sVs)[0]))
    m,n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],kTup)
        predict = kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if sign(predict)!=sign(labelArr[i]):errorCount += 1
    print('the training error rate is:%f'%(float(errorCounr)/m))

    #测试集

    dataArr,labelArr = loadImages('testDigits')
    dataMat = mat(dataArr);labelMat = mat(labelArr).transpose()
    m,n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict = kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if sign(predict)!=sign(labelArr[i]):errorCount += 1
    print('the test error rate is:%f'%(float(errorCounr)/m))



'''
>>> import svmMLiA
>>> svmMLiA.Digits()
'''






















