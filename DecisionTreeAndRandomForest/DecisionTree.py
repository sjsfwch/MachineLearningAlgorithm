#DecisionTree(CART)
import numpy as np
import collections as ct
#首先定义树的节点
class node:
    def __init__(self,feature=-1,val=None,res=None,left=None,right=None):
        self.feature=feature
        self.val=val
        self.res=res
        self.left=left
        self.right=right

class CART:
    def __init__(self,e=0.001,minSample=1):
        self.e=0.001           #阈值
        self.minSample=minSample   #每个节点最小样本数
        self.root=None     #根节点

    #计算Gini系数
    def getGini(self,trainData):   #trainData:x,y
        count=ct.Counter(trainData.y)
        return 1-sum([(val/trainData.y.shape[0])**2 for val in count.values()])  #书上公式

    #计算根据某特征分割成set1，set2后的Gini系数
    def getSplitGini(self,set1,set2):
        num=set1.shape[0]+set2.shape[0]
        return set1.shape[0] / num * self.getGini(set1) + set2.shape[0] / num * self.getGini(set2) #书上公式

    #寻找最优切分点
    def findBestSplitPoint(self,splitSet):
        G=self.getGini(splitSet)
        

    #生成树
    def buildCART(self,trainData):
