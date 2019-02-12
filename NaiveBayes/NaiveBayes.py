import numpy as np

def naiveBayesPre(trainDataMat, trainLabelMat, classNum=10, featureClassNum=2):
    # 先计算先验概率分布和条件概率分布
    featureNum = trainDataMat.shape[1]
    print(featureNum)
    Py = np.zeros((classNum, 1))
    # 求每个特征的先验概率
    for i in range(classNum):
        # 此处做了一个近似计算，防止某一类从来没出现过是0导致出现后面计算错误
        Py[i] = (np.sum(trainLabelMat == i) + 1) / (
            trainLabelMat.shape[0] + classNum)
    # 将先验概率求log，因为概率都小于1，防止后面计算的时候大量小于1的值相乘导致结果下溢
    # 而且log在定义域内是单增函数，不影响结果
    # log还可以将乘法转换为加法
    Py = np.log(Py)
    # 计算条件概率Px|y，Px_y矩阵是一个三维矩阵Px_y[i][j][k]表示当y=i时，x的第j个特征为k时的条件概率
    Px_y = np.zeros((classNum, featureNum, featureClassNum))
    # 首先计算分子
    #  featureCountList=[0]*featureClassNum
    for i in range(trainLabelMat.shape[0]):
        num = trainLabelMat[i,0]
        # print(num)
        for j in range(featureNum):
            # print(trainDataMat[i,j])
            Px_y[num][j][trainDataMat[i,j]] += 1
    # 计算分母和条件概率
    for i in range(classNum):
        for j in range(featureNum):
            sum = np.sum(Px_y[i][j])
            for k in range(featureClassNum):
                # 同样是使用贝叶斯估计而非极大似然估计
                Px_y[i][j][k] = (Px_y[i][j][k] + 1) / (sum + featureClassNum)
    Px_y = np.log(Px_y)
    return Py, Px_y


# 利用朴素贝叶斯估计概率
def naiveBayes(Py, Px_y, x, classNum=10, featureClassNum=2):
    # 根据公式计算存放所有label估计概率的数组P，因为所有概率都做了对数处理，所以乘法变加法
    P = [0] * classNum
    # print(x)
    featureNum = x.shape[1]
    for i in range(classNum):
        p = 0
        p += Py[i]
        for j in range(featureNum):
            p += Px_y[i][j][x[0,j]]
        P[i] = p
    # 找到最大概率并返回
    return P.index(max(P))