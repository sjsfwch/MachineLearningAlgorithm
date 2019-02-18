#SVM
import time
import numpy as np
import math
import random

class SVM:
    def __init__(self, trainDataList, trainLabelList, sigma = 10, C = 200, toler = 0.001):
        self.trainDataMat = np.mat(trainDataList)       #训练数据集
        self.trainLabelMat = np.mat(trainLabelList).T   #训练标签集，为了方便后续运算提前做了转置，变为列向量

        self.m, self.n = np.shape(self.trainDataMat)    #m：训练集数量    n：样本特征数目
        self.sigma = sigma                              #高斯核分母中的σ
        self.C = C                                      #惩罚参数
        self.toler = toler                              #松弛变量

        self.k = self.calcKernel()                      #核函数（初始化时提前计算）
        self.b = 0                                      #SVM中的偏置b
        self.alpha = [0] * self.trainDataMat.shape[0]   # α 长度为训练集数目
        self.E = [0 * self.trainLabelMat[i, 0] for i in range(self.trainLabelMat.shape[0])]     #SMO运算过程中的Ei
        self.supportVecIndex = []


    def calcKernel(self):
        #初始化高斯核结果矩阵 大小 = 训练集长度m * 训练集长度m
        #k[i][j] = Xi * Xj
        k = [[0 for i in range(self.m)] for j in range(self.m)]

        for i in range(self.m):
            if i % 100 == 0:
                print('construct the kernel:', i, self.m)
            #得到式7.90中的X
            X = self.trainDataMat[i, :]
            for j in range(i, self.m):
                #获得Z
                Z = self.trainDataMat[j, :]
                #先计算||X - Z||^2
                result = (X - Z) * (X - Z).T
                #分子除以分母后去指数，得到的即为高斯核结果
                result = np.exp(-1 * result / (2 * self.sigma**2))
                #将Xi*Xj的结果存放入k[i][j]和k[j][i]中
                k[i][j] = result
                k[j][i] = result
        #返回高斯核矩阵
        return k

    def isSatisfyKKT(self, i):
        gxi =self.calc_gxi(i)
        yi = self.trainLabelMat[i]
        if (math.fabs(self.alpha[i]) < self.toler) and (yi * gxi >= 1):
            return True
        #依据7.113
        elif (math.fabs(self.alpha[i] - self.C) < self.toler) and (yi * gxi <= 1):
            return True
        #依据7.112
        elif (self.alpha[i] > -self.toler) and (self.alpha[i] < (self.C + self.toler)) \
                and (math.fabs(yi * gxi - 1) < self.toler):
            return True

        return False

    def calc_gxi(self, i):
        gxi = 0
        index = [i for i, alpha in enumerate(self.alpha) if alpha != 0]
        for j in index:
            gxi += self.alpha[j] * self.trainLabelMat[j] * self.k[j][i]
        gxi += self.b
        return gxi

    def calcEi(self, i):
        gxi = self.calc_gxi(i)
        #Ei = g(xi) - yi,直接将结果作为Ei返回
        return gxi - self.trainLabelMat[i]

    def getAlphaJ(self, E1, i):
        #初始化E2
        E2 = 0
        #初始化|E1-E2|为-1
        maxE1_E2 = -1
        #初始化第二个变量的下标
        maxIndex = -1
        nozeroE = [i for i, Ei in enumerate(self.E) if Ei != 0]
        #对每个非零Ei的下标i进行遍历
        for j in nozeroE:
            #计算E2
            E2_tmp = self.calcEi(j)
            #如果|E1-E2|大于目前最大值
            if math.fabs(E1 - E2_tmp) > maxE1_E2:
                #更新最大值
                maxE1_E2 = math.fabs(E1 - E2_tmp)
                #更新最大值E2
                E2 = E2_tmp
                #更新最大值E2的索引j
                maxIndex = j
        #如果列表中没有非0元素了（对应程序最开始运行时的情况）
        if maxIndex == -1:
            maxIndex = i
            while maxIndex == i:
                #获得随机数，如果随机数与第一个变量的下标i一致则重新随机
                maxIndex = int(random.uniform(0, self.m))
            #获得E2
            E2 = self.calcEi(maxIndex)

        #返回第二个变量的E2值以及其索引
        return E2, maxIndex

    def train(self, iter = 100):
        iterStep = 0; parameterChanged = 1
        while (iterStep < iter) and (parameterChanged > 0):
            #打印当前迭代轮数
            print('iter:%d:%d'%( iterStep, iter))
            #迭代步数加1
            iterStep += 1
            #新的一轮将参数改变标志位重新置0
            parameterChanged = 0
            #大循环遍历所有样本，用于找SMO中第一个变量
            for i in range(self.m):
                #查看第一个遍历是否满足KKT条件，如果不满足则作为SMO中第一个变量从而进行优化
                if self.isSatisfyKKT(i) == False:
                    E1 = self.calcEi(i)
                    E2, j = self.getAlphaJ(E1, i)
                    y1 = self.trainLabelMat[i]
                    y2 = self.trainLabelMat[j]
                    #复制α值作为old值
                    alphaOld_1 = self.alpha[i]
                    alphaOld_2 = self.alpha[j]
                    #依据标签是否一致来生成不同的L和H
                    if y1 != y2:
                        L = max(0, alphaOld_2 - alphaOld_1)
                        H = min(self.C, self.C + alphaOld_2 - alphaOld_1)
                    else:
                        L = max(0, alphaOld_2 + alphaOld_1 - self.C)
                        H = min(self.C, alphaOld_2 + alphaOld_1)
                    #如果两者相等，说明该变量无法再优化，直接跳到下一次循环
                    if L == H:   continue
                    k11 = self.k[i][i]
                    k22 = self.k[j][j]
                    k21 = self.k[j][i]
                    k12 = self.k[i][j]
                    #依据式7.106更新α2，该α2还未经剪切
                    alphaNew_2 = alphaOld_2 + y2 * (E1 - E2) / (k11 + k22 - 2 * k12)
                    #剪切α2
                    if alphaNew_2 < L: alphaNew_2 = L
                    elif alphaNew_2 > H: alphaNew_2 = H
                    #更新α1，依据式7.109
                    alphaNew_1 = alphaOld_1 + y1 * y2 * (alphaOld_2 - alphaNew_2)

                    #依据“7.4.2 变量的选择方法”第三步式7.115和7.116计算b1和b2
                    b1New = -1 * E1 - y1 * k11 * (alphaNew_1 - alphaOld_1) \
                            - y2 * k21 * (alphaNew_2 - alphaOld_2) + self.b
                    b2New = -1 * E2 - y1 * k12 * (alphaNew_1 - alphaOld_1) \
                            - y2 * k22 * (alphaNew_2 - alphaOld_2) + self.b

                    #依据α1和α2的值范围确定新b
                    if (alphaNew_1 > 0) and (alphaNew_1 < self.C):
                        bNew = b1New
                    elif (alphaNew_2 > 0) and (alphaNew_2 < self.C):
                        bNew = b2New
                    else:
                        bNew = (b1New + b2New) / 2

                    #将更新后的各类值写入，进行更新
                    self.alpha[i] = alphaNew_1
                    self.alpha[j] = alphaNew_2
                    self.b = bNew

                    self.E[i] = self.calcEi(i)
                    self.E[j] = self.calcEi(j)
                    if math.fabs(alphaNew_2 - alphaOld_2) >= 0.00001:
                        parameterChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iterStep, i, parameterChanged))
        for i in range(self.m):
            #如果α>0，说明是支持向量
            if self.alpha[i] > 0:
                #将支持向量的索引保存起来
                self.supportVecIndex.append(i)

    def calcSinglKernel(self, x1, x2):
        result = (x1 - x2) * (x1 - x2).T
        result = np.exp(-1 * result / (2 * self.sigma ** 2))
        return np.exp(result)


    def predict(self, x):
        result = 0
        for i in self.supportVecIndex:
            tmp = self.calcSinglKernel(self.trainDataMat[i, :], np.mat(x))
            result += self.alpha[i] * self.trainLabelMat[i] * tmp
        result += self.b
        return np.sign(result)
