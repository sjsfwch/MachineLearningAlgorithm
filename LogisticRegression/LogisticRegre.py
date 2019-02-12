import numpy as np

def logisticRegression(trainDataArr,trainLabelArr,maxIterNum=188,step=0.001):
    #根据书上公式，在训练集最后加1，即在数组后面加一行全是1的列，以便把b也囊括进w
    temp=np.ones(trainDataArr.shape[0])
    np.c_[trainDataArr,temp]
    #初始化w
    w = np.zeros(trainDataArr.shape[1])
    #梯度上升
    for i in range(maxIterNum):
        for j in range(trainDataArr.shape[0]):
            xi=trainDataArr[j]
            z=np.dot(w,xi)
            yi = trainLabelArr[j]
            w+=step*(xi * yi - (np.exp(z) * xi) / ( 1 + np.exp(z)))  #书上公式 CS-229 note1关于LR部分
    return w

