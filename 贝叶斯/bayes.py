from numpy import *


def loadDataSet():#返回（词条切分后的文档集合，类别标签）
    postingList = [['my','dog','has','flea','problem','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1]#1代表侮辱性文字，0代表正常言论
    return postingList,classVec

def createVocabList(dataSet):#创建一个包含在所有文档中的出现的不重复词的列表
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) # | 用于求两个集合的并集，或操作or
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):#输入（词汇表，某个文档），输出文档向量
    returnVec = [0]*len(vocabList) # *表示and,创建一个和词汇表等长的向量，并将其元素设置为0
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1#如果出现了词汇表中的单词，将输出的文档向量中的对应值设为1
        else: print("the word:%s is not in my Vocabulary!"%word)
    return returnVec

'''
>>> import bayes
>>> listOposts,listClasses = bayes.loadDataSet()
>>> listOPosts,listClasses = bayes.loadDataSet()
>>> listOPosts
[['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'], ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'], ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'], ['stop', 'posting', 'stupid', 'worthless', 'garbage'], ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'], ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
>>> listClasses
[0, 1, 0, 1,0, 1]
>>> myVocabList = bayes.createVocabList(listOPosts)
>>> myVocabList
['please', 'quit', 'posting', 'buying', 'I', 'my', 'stupid', 'help', 'park', 'not', 'problem', 'dog', 'dalmation', 'to', 'take', 'flea', 'ate', 'licks', 'has', 'so', 'cute', 'mr', 'love', 'stop', 'maybe', 'worthless', 'food', 'is', 'garbage', 'him', 'how', 'steak']
>>> bayes.setOfWords2Vec(myVocabList,listOPosts[0])
[1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
>>> bayes.setOfWords2Vec(myVocabList,listOPosts[3])
[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0]
'''

#从词向量计算概率(文档矩阵，文档类别标签向量)
#主要是计算p(c1),p(c2),p(w1|c1)p(w2|c2)
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)#文档数
    numWords = len(trainMatrix[0])#文档中的单词数
    pAbusive = sum(trainCategory)/float(numTrainDocs)#计算文档束语侮辱性文档的概率P(1)
    
    #p0Num = zeros(numWords)#类别为0的文档中每个单词在词汇表中出现的次数向量
    #p1Num = zeros(numWords)#类别为1的文件中每个单词在词汇表中出现的次数向量
    #p0Denom = 0.0#类别为0的文档且出现在词汇表中的单词总数
    #p1Denom = 0.0#类别为1的文档且出现在词汇表中的单词总数

    #因为要计算多个概率的乘积，如果其中一个概率为0，最后乘积也为0，所有将所有词的出现数初始化为1，将分母初始化为2

    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0 
    p1Denom = 2.0

    #还有一个下溢出问题，因为大部分因子太效，所以程序会下溢出或者得不到正确的答案
    #尝试取对数，

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:#如果该词条向量对应的标签为1
            p1Num += trainMatrix[i]#统计所有类别为1的词条向量中各个词条出现的次数
            #print(p1Num)
            p1Denom += sum(trainMatrix[i])#统计类别为1的词条向量中出现的所有词条出现的总数
        else:
            p0Num += trainMatrix[i]#统计所有类别为0的词条向量中各个词条出现的次数
            p0Denom += sum(trainMatrix[i])#统计类别为0的词条向量中出现的所有词条出现的总数

    #p1Vect = p1Num/p1Denom#1类别中每个单词的数目/1类别中单词的总数目
    #p0Vect = p0Num/p0Denom#0类别中每个单词出现的次数/0类别中单词的总数目
    #还有一个下溢出问题，因为大部分因子太效，所以程序会下溢出或者得不到正确的答案
    #尝试取对数
    #利用numpy数组计算p(wi|c)
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)

    return p0Vect,p1Vect,pAbusive
'''
#>>> from numpy import *
#>>> import bayes
#>>> listOPosts,listClasses = bayes.loadDataSet()
>>> listOPosts
[['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'], ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'], ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'], ['stop', 'posting', 'stupid', 'worthless', 'garbage'], ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'], ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
>>> listClasses
[0, 1, 0, 1, 0, 1]
>>> myVocabList = bayes.createVocabList(listOPosts)
>>> myVocabList
['I', 'quit', 'mr', 'how', 'so', 'dog', 'dalmation', 'flea', 'worthless', 'my', 'not', 'stupid', 'cute', 'is', 'please', 'steak', 'maybe', 'has', 'park', 'love', 'garbage', 'posting', 'ate', 'food', 'problem', 'licks', 'stop', 'take', 'help', 'him', 'to', 'buying']
>>> trainMat = []
>>> for postinDoc in listOPosts:
	trainMat.append(bayes.setOfWords2Vec(myVocabList,postinDoc))	
>>> trainMat
[[0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0], [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0], [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]]
>>> len(myVocabList)
32
>>> len(trainMat[0])
32 

>>> p0V,p1V,pAb = bayes.trainNB0(trainMat,listClasses)
>>> p0V
array([0.04166667, 0.04166667, 0.        , 0.04166667, 0.04166667,
       0.125     , 0.04166667, 0.04166667, 0.04166667, 0.04166667,
       0.04166667, 0.08333333, 0.        , 0.04166667, 0.        ,
       0.        , 0.        , 0.        , 0.04166667, 0.04166667,
       0.04166667, 0.        , 0.        , 0.        , 0.04166667,
       0.04166667, 0.        , 0.04166667, 0.04166667, 0.        ,
       0.04166667, 0.04166667])
>>> p1V
array([0.        , 0.        , 0.05263158, 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.05263158, 0.05263158, 0.05263158, 0.05263158,
       0.05263158, 0.05263158, 0.05263158, 0.05263158, 0.        ,
       0.10526316, 0.15789474, 0.05263158, 0.05263158, 0.        ,
       0.        , 0.10526316, 0.        , 0.        , 0.05263158,
       0.        , 0.        ])
>>> pAb
0.5
>>> len(trainMat)
6
>>> trainMat[0]
[0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
>>> sum(trainMat[0])
7
>>> trainMat[1]
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
>>> trainMat[3]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
>>> trainMat[5]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]
>>> trainMat[1]+trainMat[3]+trainMat[5]
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]
>>> sum(trainMat[1])+sum(trainMat[3])+sum(trainMat[5])
19

'''
#朴素贝叶斯分类函数
#类别0所有文档中各个词条出现的频率p(wi|c0)
#类别1所有文档中各个词条出现的频率p(wi|c1)
#p(c1),p(c0)，类别c1，c0的文档栈文档总数比例
#p(c0|wi)=(p(wi|c0)*p(c0))/p(w0)
#p(c0|wi)=(p(wi|c0)*p(c0))/p(w0)


'''
将惩罚转换为加法
乘法：p(c|F1F2...Fn) = p(F1F2...Fn|c)p(c)/p(F1F2...Fn)
加法：p(F1|c)*p(F2|c)...p(Fn|c)p(c) ->log(p(F1|c))+log(p(F2|c))+...+log(p(Fn|c))+log(p(c))
p0Vec:类别0，正常文档中，各个单词出现的概率列表
p1Vec:类别1，侮辱性文档中，各个单词出现的概率列表
pClass1:侮辱性文件出现的概率


'''
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify*p1Vec) + log(pClass1)
    #p(w1|c1)*p(w2|c1)*...*p(wn|c1)*p(c1)，利用朴素贝叶斯分别计算待分类文档属于类0和类1的概率
    #print(vec2Classify)
    #print(p1Vec)
    #print(vec2Classify*p1Vec)
    p0 = sum(vec2Classify*p0Vec) + log(1.0 - pClass1)
    ##p(w1|c0)*p(w2|c0)*...*p(wn|c0)*p(c0)
    if(p1>p0):
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()#由数据集获取文档矩阵和类标签向量
    myVocabList = createVocabList(listOPosts)#统计所有文档中出现的词条，存入词条列表
    trainMat = []
    for postinDoc in listOPosts:#将每篇文档利用words2Vec函数转为词向量，存入文档矩阵
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))#训练文档，得到相应的概率
    testEntry = ['love','my','dalmation']#测试文档
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))#将测试文档转为词条向量
    print(testEntry,'classified as :',classifyNB(thisDoc,p0V,p1V,pAb))#利用贝叶斯对测试文档进行分类
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as :',classifyNB(thisDoc,p0V,p1V,pAb))

#过滤垃圾邮件
#1、对长字符串进行分割，分隔符为除单词和数字之外的任意符号串
#2、将分割后的字符串中所有的大写字母变成小写Lower(),并且只百六单词长度大于3的单词

def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        returnVec[vocabList.index(word)] += 1
    return returnVec

#切分文本，string.split()
#
'''
>>> mySent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
>>> bayes.textParse(mySent)
['this', 'book', 'the', 'best', 'book', 'python', 'have', 'ever', 'laid', 'eyes', 'upon']
>>> mySent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
>>> mySent.split()
['This', 'book', 'is', 'the', 'best', 'book', 'on', 'Python', 'or', 'M.L.', 'I', 'have', 'ever', 'laid', 'eyes', 'upon.']
>>> import re
>>> regEx = re.compile('\\W')
>>> listOfTokens = regEx.split(mySent)
>>> listOfTokens
['This', 'book', 'is', 'the', 'best', 'book', 'on', 'Python', 'or', 'M', 'L', '', 'I', 'have', 'ever', 'laid', 'eyes', 'upon', '']
>>> [tok for tok in listOfTokens if len(tok) > 0]
['This', 'book', 'is', 'the', 'best', 'book', 'on', 'Python', 'or', 'M', 'L', 'I', 'have', 'ever', 'laid', 'eyes', 'upon']
>>> [tok.lower() for tok in listOfTokens if len(tok) > 0]
['this', 'book', 'is', 'the', 'best', 'book', 'on', 'python', 'or', 'm', 'l', 'i', 'have', 'ever', 'laid', 'eyes', 'upon']
>>> emailText = open('email/ham/6.txt').read()
>>> listOfTokens = regEx.split(emailText)
>>> listOfTokens

'''

def textParse(bigString):#输入字符串，切分字符串列表
    import re
    listOfTokens = re.split(r'\W',bigString)
    #python3中'\\W'表示分隔符是除字母数字以外的其他字符串
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]#返回字符串长度大于1的小写字符串
'''
>>> import bayes
>>>  mySent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
 
SyntaxError: unexpected indent
>>> mySent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
>>> bayes.textParse(mySent)
['this', 'book', 'the', 'best', 'book', 'python', 'have', 'ever', 'laid', 'eyes', 'upon']
'''

#对贝叶斯垃圾邮件分类器进行自动化处理
def spamTest():
    docList = [];classList = [];fullText = []
    for i in range(1,26):#1是垃圾邮件，0是正常邮件
        wordList = textParse(open('email/spam/%d.txt'%i,encoding='Shift_JIS').read())#导入并解析25个文本文件成字符串列表
        docList.append(wordList)#把多个列表分组添加到一个列表中
        fullText.extend(wordList)#把多个列表添加到一个列表中
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt'%i,encoding='Shift_JIS').read())#导入并解析25个文本文件成字符串列表
        docList.append(wordList)#把多个列表分组添加到一个列表中
        fullText.extend(wordList)#把多个列表添加到一个列表中
        classList.append(0)

    vocabList = createVocabList(docList)#生成词汇表
    trainingSet = list(range(50));testSet = []#trainingSet是暂且赋值为从0到49的列表
    for i in range(10):#生成一个长为10的随机列表作为测试集合,随机选出10个测试集合
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])#把选作测试的集合从训练集合中删除
    trainMat = [];trainClasses = []
    for docIndex in trainingSet:#剩余40个的数据用来训练分类器，得到概率p(wi|c0)，p(wi|c1)，p(c1),p(c0)
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))#训练分类器
    errorCount = 0
    for docIndex in testSet:#分类测试数据
        wordVector = bagOfWords2VecMN(vocabList,docList[docIndex])#把测试数据集转化为数字向量
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print('classification error',docList[docIndex],)
            print('正确分类为',classList[docIndex])
            print('分类结果',classifyNB(array(wordVector),p0V,p1V,pSpam))
    print('the error rate is:',float(errorCount)/len(testSet))

#>>> import bayes
#>>> bayes.spamTest() 
'''        
if __name__ == '__main__':
    spamTest()
'''

'''
先下载feedparser
在cmd中进入feedparser的文件夹，运行 python setup.py install
>>> import feedparser
>>> ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
>>> ny['entries']
[]
>>> len(ny['entries'])
0
'''

#RSS源分类器及高频词去除函数

def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.item(),key=operator.itemgetter(1),reverse=True)
    return sortedFreq[:30]

def localWords(feed1,feed0):
    import feedparser
    docList = [];classList = [];fullText = []
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top3Words = calcMostFreq(vocabList,fullText)
    for pairw in top3Words:
        if pairW[0] in vocabList:vocabList.remove(pairW[0])
    trainingSet = range(2*minLen);testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = [];trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) !=classList[docIndex]:
            errorCount += 1
    print('the erroe rate is:',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

'''
网站可能关闭，entries为空集，无法测试
>>> import bayes
>>> ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
>>> sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
>>> vocabList,pSF,pNY = bayes.localWords(ny,sf)
>>> vocabList,pSF,pNY = bayes.localWords(ny,sf)
'''

def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V = localWords(ny,sf)
    topNY = [];topSF = []
    for i in range(p0V):
        if p0V[i] > -6.0:topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0:topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF,key=lambda pair:pair[1],reverse=True)
    print('SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**')
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY,key=lambda pair:pair[1],reverse=True)
    print('NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**')
    for item in sortedNY:
        print item[0]

'''
>>> import bayes
>>> ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
>>> sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
>>> bayes.getTopWords(ny,sf)
'''
















    











    
