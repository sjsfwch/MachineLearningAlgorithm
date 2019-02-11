import numpy as np

def kNN(dataMat,labelMat,x,k):
    distList=[]
    for i in range(dataMat.shape[0]):
        xi=dataMat[i]
        dist=np.sqrt(np.sum(np.square(xi - x)))
        distList.append(dist)
    kList=np.argsort(np.array(distList))[:k]
    labelList=[0 for i in range(10)]
    for i in kList:
        labelList[labelMat[i]]+=1
    return labelList.index(max(labelList))

