from math import log
import operator

#计算熵,熵越大，数据越混乱，不确定性越大
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)#获取数据集的长度
    labelCounts = {}#字典统计
    for featVec in dataSet:#遍历特征
        currentLabel = featVec[-1]#最后一维特征（类别）
        if currentLabel not in labelCounts.keys():#如果当前类别不在字典中，添加进字典
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1#统计每个类别的数量
    #熵
    shannonEnt = 0.0
    for key in labelCounts:#对于每一个类别遍历
        prob = float(labelCounts[key])/numEntries#该分类出现的次数除以总的分类次数
        shannonEnt -= prob * log(prob,2)#计算熵
    return shannonEnt

#测试
def createDataSet():
    dataSet = [
        [1,1,'yes'],
        [1,1,'yes'],
        [1,0,'no'],
        [0,1,'no'],
        [0,1,'no']  ]
    labels = ['no surfacing','flippers']
    return dataSet,labels

#>>> import trees
#>>> myDat,labels = trees.createDataSet()
#>>> myDat
#>>> labels
#>>> trees.calcShannonEnt(myDat)
#-2/5*log2(2/5)-3/5*log2(3/5)=0.9709505944546686
#>>> myDat[0][-1] = 'maybe'
#>>> trees.calcShannonEnt(myDat)

#划分数据集（待划分的数据集，划分数据集的特征，特征的返回值）
#注意，python函数传递的是列表的引用，所以修改列表将影响列表整个生命周期

#思想是去除特征axis的取值等于value的值，因为如果以这个数据集划分，
#相当于这个特征已经被用了，所以去除该特征的相关值，得到剩下的特征的数据，
#方便进行下一步的划分
def splitDataSet(dataSet,axis,value):
    retDataSet = []#返回数据集
    for featVec in dataSet:#遍历dataSet，去掉value
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]#从0赋值到axis-1
            #print(reducedFeatVec)
            reducedFeatVec.extend(featVec[axis+1:])#从axis+1赋值到len(myDat[0])
            #print(reducedFeatVec)
            retDataSet.append(reducedFeatVec)#把去掉value值的其他数据按axis特征进行数据划分
    return retDataSet
#>>> import trees
#>>> myDat,labels = trees.createDataSet()
#>>> trees.splitDataSet(myDat,0,1)
 
'''
>>> a=[1,2,3]
>>> b=[4,5,6]
>>> a.append(b)
>>> a
[1, 2, 3, [4, 5, 6]]
>>> a=[1,2,3]
>>> a.extend(b)
>>> a
[1, 2, 3, 4, 5, 6]

'''

#选择最好的数据集划分方式,计算出最好的划分数据集的特征

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1 #特征的数目，最后一列是结果
    baseEntropy = calcShannonEnt(dataSet)#计算经验熵
    bestInfoGain = 0.0;bestFeature = -1
    for i in range(numFeatures):#遍历每一个feature
        featList = [example[i] for example in dataSet]#提取出这个特征
        uniqueVals = set(featList)#python集合，目的是去除重复，获得这个特征所有的取值范围
        newEntropy = 0.0
        for value in uniqueVals:#遍历这个特征的取值范围
            subDataSet = splitDataSet(dataSet,i,value)#根据某个特征取值，找出子数据集
            prob = len(subDataSet)/float(len(dataSet))#计算经验条件熵
            newEntropy += prob*calcShannonEnt(subDataSet)#概率*这个特征为value时候的熵
            #print(newEntropy) 
            infoGain = baseEntropy - newEntropy#信息增益
            #print(infoGain)
            if(infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeature = i
    return bestFeature

#>>> import trees
#>>> myDat,labels = trees.createDataSet()
#>>> trees.chooseBestFeatureToSplit(myDat)
'''
0.0
0.9709505944546686
0.5509775004326937
0.4199730940219749大
0.0
0.9709505944546686
0.8
0.17095059445466854小
0
'''
#多数表决，进行投票表决，类似classify0部分代码

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.key():classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#创建决策树代码（数据集、标签列表）     

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]#获取dataSet分类
    if classList.count(classList[0]) == len(classList):
        return classList[0]#类别完全相同则停止继续划分
    if len(dataSet[0]) == 1:#特征全部用完，剩余最后的分类标签
        return majorityCnt(classList)#遍历完所有特征时返回出现次数最多的类别
    bestFeat = chooseBestFeatureToSplit(dataSet)#返回最大信息增益的特征--0
    bestFeatLabel = labels[bestFeat]#特征到标签--no surfacing
    myTree = {bestFeatLabel:{}}#构建一个树，先生成根,以no surfacing
    del(labels[bestFeat])#去掉labels中下标是bestFeat的元素
    featValues = [example[bestFeat] for example in dataSet]#取一列
    uniqueVals = set(featValues)#集合，去重
    for value in uniqueVals :
        subLabels = labels[:]#浅拷贝
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)#在孩子节点上建立子树
    return myTree#返回决策树

#>>> import trees
#>>> myDat,labels=trees.createDataSet()
#>>> myTree=trees.createTree(myDat,labels)
#>>> myTree


#测试，使用决策树执行分类算法

def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else: classLabel = secondDict[key]
    return classLabel
            
'''
>>> import trees
>>> myDat,labels = trees.createDataSet()
>>> myDat
[[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
>>> labels
['no surfacing', 'flippers']
>>> import treePlotter
>>> myTree = treePlotter.retrieveTree(0)
>>> myTree
{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
>>> trees.classify(myTree,labels,[1,0])
'no'
>>> trees.classify(myTree,labels,[1,1])
'yes'

'''

#存储决策树

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)

'''
>>> import trees
>>> myDat,labels = trees.createDataSet()
>>> import treePlotter
>>> myTree = treePlotter.retrieveTree(0)
>>> trees.storeTree(myTree,'classifierStorage.txt')
>>> trees.grabTree('classifierStorage.txt')
{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}

'''

#测试隐形眼镜类型
'''
>>> fr = open('lenses.txt')
>>> lenses = [inst.strip().split('\t') for inst in fr.readlines()]
>>> lensesLabels = ['age','prescript','astigmatic','tearRate']
>>> import trees
>>> import treePlotter
>>> lensesTree = trees.createTree(lenses,lensesLabels)
>>> lensesTree
{'tearRate': {'reduced': 'no lenses', 'normal': {'astigmatic': {'no': {'age': {'young': 'soft', 'presbyopic': {'prescript': {'myope': 'no lenses', 'hyper': 'soft'}}, 'pre': 'soft'}}, 'yes': {'age': {'young': 'hard', 'presbyopic': {'prescript': {'myope': 'hard', 'hyper': 'no lenses'}}, 'pre': {'prescript': {'myope': 'hard', 'hyper': 'no lenses'}}}}}}}}
>>> treePlotter.createPlot(lensesTree)
'''


#测试决策树分类性能'the calculte result is%s,the true result is %s'%(result,testdata[i][-1])
def testefficiency(inputtree,labels,testdata):
    flag=0
    for i in range(len(testdata)):
        result=classify(inputtree,labels,testdata[i][:2])
        if(result==testdata[i][-1]):
            flag +=1
            print('the calculte result is%s,the true result is %s'%(result,testdata[i][-1]))
    print('the data number is %d,but the right number is %d'%(len(testdata),flag))

'''
>>> import trees
>>> myDat,labels = trees.createDataSet()
>>> import treePlotter
>>> myTree = treePlotter.retrieveTree(0)
>>> trees.testefficiency(myTree,labels,[[1,0,'no'],[1,1,'yes']])
the calculte result isno,the true result is no
the calculte result isyes,the true result is yes
the data number is 2,but the right number is 2

'''
