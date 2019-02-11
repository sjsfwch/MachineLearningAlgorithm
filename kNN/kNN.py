import numpy as np

def kNN(dataMat,labelMat,x,k):
    distList=[]            #用于存储每个数据到x的距离（本算法采用的欧式距离）
    for i in range(dataMat.shape[0]):    #计算训练集中的数据与x的距离
        xi=dataMat[i]
        dist=np.sqrt(np.sum(np.square(xi - x)))
        distList.append(dist)
    kList=np.argsort(np.array(distList))[:k]    #取与x最近的k个数据
    labelList=[0 for i in range(10)]
    for i in kList:                        #找出出现次数最多的那个label
        labelList[labelMat[i]]+=1
    return labelList.index(max(labelList))

