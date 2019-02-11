import numpy as np


def perceptron(dataArray, labelArray, maxIterNum=66, step=0.01):
    print('start!!')
    dataMat = np.mat(dataArray)  #转换成矩阵，运算比较方便
    labelMat = np.mat(labelArray).T
    m, n = np.shape(dataMat)
    w = np.zeros((1, n))  #初始化w,b参数
    b = 0
    for i in range(maxIterNum):
        print('Round %d' % i)
        flag = True  #用于判断是否已经找到超平面
        errCount = 0
        for j in range(m):
            x = dataMat[j]
            y = labelMat[j]
            if -1 * y * (w * x.T + b) >= 0:
                flag = False
                errCount += 1
                w = w + step * y * x
                b = b + step * y
        print('accuracy',1-errCount/m)  #该轮训练的正确率
        if (flag):  #找到完成分割的超平面，跳出循环
            break
    return w, b
