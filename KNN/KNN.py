from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir
import numpy as np
import matplotlib as mpl
import matplotlib.lines as mlines

'''
KNN是一种最简单最有效的算法，但是KNN必须保留所有的数据集，
如果训练数据集很大，必须使用大量的存储空间，
此外，需要对每一个数据计算距离，非常耗时
另外，它无法给出任何数据的基础结构信息(无法给出一个模型)
'''

#使用python导入数据
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

#实施KNN分类算法
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]#查看矩阵的维度
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    #tile(数组,(在行上重复次数,在列上重复次数))
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    #sum默认axis=0，是普通的相加，axis=1是将一个矩阵的每一行向量相加
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    #sort函数按照数组值从小到大排序
    #argsort函数返回的是数组值从小到大的索引值
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        #get(key,k),当字典dic中不存在key时，返回默认值k;存在时返回key对应的值
    sortedClassCount = sorted(classCount.items(),
           key=operator.itemgetter(1),reverse=True)
    #python2中用iteritems，python3中用items代替；operator.itemgetter(k),返回第k个域的值
    return sortedClassCount[0][0]
#测试KNN
#>>> import KNN
#>>> group,labels = KNN.createDataSet()
#>>> KNN.classify0([0,0],group,labels,3)
#>>> KNN.classify0([1.2,1.5],group,labels,3)

#读取数据
def file2matrix(filename):
    fr = open(filename) #打开文件
    arrayOLines = fr.readlines() #读取文件所有内容
    numberOfLines = len(arrayOLines) #得到文件行数
    returnMat = np.zeros((numberOfLines,3)) #返回numpy矩阵，解析完成的数据：numberOfLines行，3列
    classLabelVector = []#返回的分类标签向量
    index = 0 #行的索引值
    for line in arrayOLines:
        line = line.strip() #删除空白符'\n','\r','\t',''
        listFromLine = line.split('\t')#将字符串根据'\t'分隔符进行切片
        returnMat[index,:] = listFromLine[0:3]#将数据前三列提取出来，存放在returnMat的numpy矩阵中，也就是特征矩阵
        #classLabelVector.append(int(listFromLine[-1]))#根据文本中标记的喜欢程度进行分类
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index+=1
    return returnMat,classLabelVector
#datingDataMat,datingLabels = KNN.file2matrix('datingTestSet2.txt')
'''
datingTestSet.txt和datingTestSet2.txt分类标签不同，
前者用的中文，后者用的英文，所以classLabelVector那里写的不一样
            if listFromLine[-1] == 'didntLike':

                classLabelVector.append(1)

            elif listFromLine[-1] == 'smallDoses':

                classLabelVector.append(2)

            elif listFromLine[-1] == 'largeDoses':

                classLabelVector.append(3)
'''

#datingDataMat
#datingLabels


#不带颜色的图式
#>>> import matplotlib
#>>> import matplotlib.pyplot as plt
#>>> fig = plt.figure()
#>>> ax = fig.add_subplot(111)
#>>> ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
#<matplotlib.collections.PathCollection object at 0x00000148E4C4AE88>
#>>> plt.show()

#带颜色的图式
'''
>>> import matplotlib
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> ax = fig.add_subplot(111)
>>> import numpy as np
>>> ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*np.array(datingLabels),15.0*np.array(datingLabels))
<matplotlib.collections.PathCollection object at 0x00000148E1C5DAC8>

>>> ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*np.array(datingLabels),15.0*np.array(datingLabels))
#显示的图像更有规律性
>>> plt.show()

'''

'''
python2中reload的解决方法

>>> import importlib,sys
>>> importlib.reload(sys)
<module 'sys' (built-in)>
python3的解决办法
>>> import importlib,sys
>>> importlib.reload(sys)
<module 'sys' (built-in)>
'''

#分析数据，数据可视化，使用matplotlib创建散点图

def showdatas(datingDataMat,datingLabels):
    #设置汉字格式
    #sans-serif就是无衬线字体，是一种通用字体族
    mpl.rcParams['font.sans-serif'] = ['SimHei']#指定默认字体，SimHei为黑体
    mpl.rcParams['axes.unicode_minus'] = False #用来正常显示负号
    #将fig画布分隔成2行2列，不共享x轴和y轴，fig画布的大小为（13，8）
    #当nrow=2,nclos=2时，代表fig画布被分成四个区域，axs[0][0]表示第一行第一个区域
    fig,axs = plt.subplots(nrows=2,ncols=2,sharex=False,sharey=False,figsize=(13,10))
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    #画出散点图，以datingDataMat矩阵的第一（飞行常客例程）、第二列（玩游戏）数据画散点数据，散点大小为15，透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:,0],y=datingDataMat[:,1],color=LabelsColors,s=15,alpha=0.5)
    #设置标题，x轴label,y轴label
    axs0_title_text = axs[0][0].set_title('每年获得的飞行常客里程数与玩视频游戏所消耗时间占比')
    axs0_xlabel_text = axs[0][0].set_xlabel('每年获得的飞行常客里程数')
    axs0_ylabel_text = axs[0][0].set_ylabel('玩视频游戏所消耗时间占比')
    plt.setp(axs0_title_text,size=8,weight='bold',color='red')
    plt.setp(axs0_xlabel_text,size=6,weight='bold',color='black')
    plt.setp(axs0_xlabel_text,size=6,weight='bold',color='black')
    

    #画出散点图，以datingDataMat矩阵的第一（飞行常客例程）、第二列（冰激凌）数据画散点数据，散点大小为15，透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:,0],y=datingDataMat[:,2],color=LabelsColors,s=15,alpha=0.5)
    #设置标题，x轴label,y轴label
    axs1_title_text = axs[0][1].set_title('每年获得的飞行常客里程数与每周消费的冰激凌公升数')
    axs1_xlabel_text = axs[0][1].set_xlabel('每年获得的飞行常客数')
    axs1_ylabel_text = axs[0][1].set_ylabel('每周消费的冰激凌公升数')
    plt.setp(axs1_title_text,size=8,weight='bold',color='red')
    plt.setp(axs1_xlabel_text,size=6,weight='bold',color='black')
    plt.setp(axs1_xlabel_text,size=6,weight='bold',color='black')

    #画出散点图，以datingDataMat矩阵的第一（玩游戏）、第二列（冰激凌）数据画散点数据，散点大小为15，透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:,1],y=datingDataMat[:,2],color=LabelsColors,s=15,alpha=0.5)
    #设置标题，x轴label,y轴label
    axs2_title_text = axs[1][0].set_title('玩视频游戏所消耗时间占比与每周消费的冰激凌公升数')
    axs2_xlabel_text = axs[1][0].set_xlabel('玩视频游戏所消耗时间占比')
    axs2_ylabel_text = axs[1][0].set_ylabel('每周消费的冰激凌公升数')
    plt.setp(axs2_title_text,size=8,weight='bold',color='red')
    plt.setp(axs2_xlabel_text,size=6,weight='bold',color='black')
    plt.setp(axs2_xlabel_text,size=6,weight='bold',color='black')

    #设置图例
    didntLike = mlines.Line2D([],[],color='black',marker='.',markersize=6,label='不喜欢')
    smallDoses = mlines.Line2D([],[],color='orange',marker='.',markersize=6,label='有点喜欢')
    largeDoses = mlines.Line2D([],[],color='red',marker='.',markersize=6,label='非常喜欢')

    #添加图例
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])

    #显示图片
    plt.show()
    

#规一化数值
def autoNorm(dataSet):
    #获得每列数据的最小值和最大值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    #shape(dataSet)返回dataSet的矩阵行列数
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet .shape[0]#返回dataSet的行数
    normDataSet = dataSet-np.tile(minVals,(m,1))#原始值减去最小值
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals
#>>> import KNN
#>>> datingDataMat,datingLabels = KNN.file2matrix('datingTestSet2.txt')
#>>> normMat,ranges,minVals = KNN.autoNorm(datingDataMat)
#>>> normMat
#>>> ranges
#>>> minVals

#测试分类器错误率

def datingClassTest():
    heRatio = 0.10#取所有数据的百分之十
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')#读取数据
    normMat,ranges,minVals = autoNorm(datingDataMat)#进行归一化
    m = normMat.shape[0]#返回normMat的行数
    numTestVecs = int(m*heRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        #前numTestVecs个数据作为测试集，后m-numTestVecs个数据作为训练集
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],4)
        print("the classifierResult came back with:%d,the real answer is:%d"%(classifierResult,datingLabels[i]))
        if(classifierResult !=datingLabels[i]):
            errorCount += 1.0
        
    print("the total error rate is:%f"%(errorCount/float(numTestVecs)))

#>>> import KNN
#>>> KNN.datingClassTest()
    
#构造完整可用系统
def classifyPerson():
    #输出结果
    resultList = ['不喜欢','有点喜欢','非常喜欢']
    #三维特征用户输入
    ffMiles = float(input("每年获得的飞行常客里程数："))
    precentTats = float(input("完视频游戏所耗时间百分比："))
    iceCream = float(input("每周消费的冰激凌公升数："))
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')#打开并处理数据
    normMat,ranges,minVals = autoNorm(datingDataMat)#训练集归一化
    inArr = np.array([ffMiles,precentTats,iceCream])#生成Numpy数组，测试集
    norminArr = (inArr - minVals)/ranges#测试集归一化
    classifierResult = classify0(norminArr,normMat,datingLabels,4)#返回分类结果
    print("你可能%s这个人"%(resultList[classifierResult-1]))

#>>> import KNN
#>>> KNN.classifyPerson()

#主函数，测试以上各个步骤，并输出各个步骤的结果
if __name__ == '__main__':
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')#打开处理文件
    showdatas(datingDataMat,datingLabels)#数据可视化
    datingClassTest()#验证分类器
    classifyPerson()#使用分类器



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

#>>> import KNN
#>>> testVector = KNN.img2vector('testDigits/0_13.txt')
#>>> testVector[0,0:31]
#>>> testVector[0,32:63]

def handwritingClassTest():
    hwLabels = []#定义一个list，用于记录分类
    trainingFileList = listdir('trainingDigits')#获取训练数据集的目录
    #os.listdir可以列出dir里面的所有文件和目录，但不包括子目录中的内容
    #os.walk可以遍历下面的所有目录，包括子目录
    m = len(trainingFileList)#求出文件的长度
    trainingMat = zeros((m,1024))#训练矩阵，生成m*1024的array，每个文件分配1024个0
    for i in range(m):#循环，对每个file
        fileNameStr = trainingFileList[i]#当前文件
        #9_45.txt，9代表分类，45表示第45个
        fileStr = fileNameStr.split('.')[0]#首先去掉txt
        classNumStr = int(fileStr.split('_')[0])#然后去掉_，得到分类
        hwLabels.append(classNumStr)#把分类添加到标签上
        trainingMat[i,:] = img2vector('trainingDigits/%s'%fileNameStr)#进行向量化
    testFileList = listdir('testDigits')#处理测试文件
    errorCount = 0.0#计算误差个数
    mTest = len(testFileList)#取得测试文件个数

    for k in range(1,20):#遍历不同k对错误率的影响
        errorCount = 0.0
        for i in range(mTest):#遍历测试文件
            fileNameStr = testFileList[i]
            fileStr = fileNameStr.split('.')[0]
            classNumStr = int(fileStr.split('_')[0])
            vectorUnderTest = img2vector('trainingDigits/%s'%fileNameStr)
            classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,k)
            #print('the classifier came back with:%d,the real answer is:%d'%(classifierResult,classNumStr))
            if(classifierResult != classNumStr):errorCount+=1.0
        print('\nthe total number of errors is:%d'%errorCount)
        print('\nthe total rate is:%f'%(errorCount/float(mTest)))
        print('k is {} and the correct rate is{}%'.format(k,(mTest-errorCount)*100/mTest))

    

#>>> import KNN
#>>> KNN.handwritingClassTest()

    
    


    
