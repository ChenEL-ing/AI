import matplotlib.pyplot as plt

from matplotlib import font_manager as fm,rcParams
import matplotlib as plotlib

#使用文本注解绘制树节点
#包括了边框的类型，边框线的粗细等

decisionNode = dict(boxstyle="sawtooth",fc="0.8")
#boxstyle为文本框的类型，fc是边框线粗细，pad指的是外边框锯齿形（圆形等）的大小
leafNode = dict(boxstyle="round4",fc="0.8")#定义决策树的叶子结点的描述属性，round4表示圆形
arrow_args = dict(arrowstyle="<-")#定义箭头属性

'''
annotate是关于一个数据点的文本
nodeTxt为要显示的文本，centerPt为文本的中心点，箭头所在的点，parentPt为指向文本的点
annotate的作用是添加注释，nodetxt是注释的内容
nodetype指的是输入的结点（边框）的形状
center_pt文本中心点，箭头指向的点，parent_pt箭头的起点
'''

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
     
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords="axes fraction",\
                            xytext=centerPt,textcoords="axes fraction",\
                            va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)
'''
#创建画板
def createPlot():
    fig = plt.figure(1,facecolor="white")#1表示第一个图，背景色为白色
    fig.clf()#清空画板
    createPlot.ax1=plt.subplot(111,frameon=False)
    #subplot(x*y*z)表示把画板分割成x*y的网格，z是画板的标号
    #frameon=False表示不绘制坐标轴矩形
    plotNode('决策节点',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode("叶节点",(0.8,0.1),(0.3,0.8),leafNode)
    plotlib.rcParams['font.sans-serif']=['SimHei']#设置显示汉字
    plotlib.rcParams['axes.unicode_minus']=False

    plt.show()
'''

'''
字体设置错误，使用下面可以显示汉字
>>> from matplotlib import font_manager as fm,rcParams
>>> import matplotlib as plt
>>> plotlib.rcParams['font.sans-serif']=['SimHei']
>>> plotlib.rcParams['axes.unicode_minus']=False
>>> import treePlotter
>>> treePlotter.createPlot()
>>> 
'''

#绘制树，首先要清楚叶子节点的数量以及树的深度，以便确定xy轴的长度
#获取叶子节点的个数
def getNumLeafs(myTree):
    numLeafs = 0
    #firstStr = myTree.keys()[0]#找到第一个节点,python2的用法
    #firstStr = next(iter(myTree)) #找到根节点
    firstStr = list(myTree.keys())[0]#'no surfacing'
    secondDict = myTree[firstStr]#{0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else: numLeafs += 1#当不是字典的时候，说明是叶节点
    return numLeafs

#得到树的深度
def getTreeDepth(myTree):
    maxDepth = 0
    #firstStr = myTree.keys()[0]
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1+getTreeDepth(secondDict[key])#是字典的时候，加1层后，继续遍历
        else: thisDepth = 1#不是字典的时候，说明到了页节点，加1，该分支结束
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth
    
#测试树叶节点个数和树深度的树数据

def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},\
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1:'yes'}}, 1: 'no'}}}} ]

    return listOfTrees[i]

#>>> import treePlotter
#>>> treePlotter.retrieveTree(1)
#>>> myTree = treePlotter.retrieveTree(0)
#>>> myTree
#>>> treePlotter.getNumLeafs(myTree)
#>>> treePlotter.getTreeDepth(myTree)


#在父子节点间填充文本信息
def plotMidText(cntrPt,parentPt,txtString):
    xMid = (parentPt[0]+cntrPt[0])/2.0#中点坐标公式
    yMid = (parentPt[1]+cntrPt[1])/2.0
    #xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    #yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString)
    #createPlot方法的ax1属性为一个plot视图，此处为视图添加文本


#计算树的宽和高
def plotTree(myTree,parentPt,nodeTxt):
    numLeafs = getNumLeafs(myTree)#获得叶节点数
    depth = getTreeDepth(myTree)#获得树的深度
    firstStr = list(myTree.keys())[0]#获取根节点
    cntrPt = (plotTree.xOff+(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
    #子节点坐标计算
    plotMidText(cntrPt,parentPt,nodeTxt)#填充父子节点的文本
    plotNode(firstStr,cntrPt,parentPt,decisionNode)#绘制树节点
    secondDict = myTree[firstStr]#通过第一个根获取value的值
    plotTree.yOff = plotTree.yOff-1.0/plotTree.totalD#树y的坐标偏移量

    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':#如果遍历到字典，将调用本身绘制子节点
            plotTree(secondDict[key],cntrPt,str(key))
        else:#已经遍历不到字典，此处已经是最后一个，将其画上
            plotTree.xOff = plotTree.xOff+1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)#绘制子节点
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))#添加节点间的文本信息
    plotTree.yOff = plotTree.yOff+1.0/plotTree.totalD#确定y的偏移量

#创建视图
def createPlot(inTree):
    fig = plt.figure(1,facecolor="white")#1表示第一个图，背景色为白色
    fig.clf()#清空画板
    axprops = dict(xticks=[],yticks=[])#不需要设置xy的刻度文本
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)
    #subplot(x*y*z)表示把画板分割成x*y的网格，z是画板的标号
    #frameon=False表示不绘制坐标轴矩形
    plotTree.totalW = float(getNumLeafs(inTree))#总的宽度等于叶子节点的数量
    plotTree.totalD = float(getTreeDepth(inTree))#总的高度等于树的层数
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree,(0.5,1.0),'')
    plotlib.rcParams['font.sans-serif']=['SimHei']#设置显示汉字
    plotlib.rcParams['axes.unicode_minus']=False

    plt.show()


'''
>>> import treePlotter
>>> myTree = treePlotter.retrieveTree(0)
>>> treePlotter.createPlot(myTree)
>>> myTree['no surfacing'][2] = 'maybe'
>>> myTree
{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}, 2: 'maybe'}}
>>> treePlotter.createPlot(myTree)
'''

#测试，使用决策树执行分类















