import matplotlib.pyplot as plt
import numpy as np
import math,random
import sys
sys.path.append("..")
from common.DataLoad import DataLoad

class SMO(object):
    def __init__(self,C,kernelopt=(0)):
        self.C=C
        self.b=0
        self.W=0
        self.maxIter=5000
        self.alpha=None

        #支持向量和支持向量的alpha
        self.supVectors=[]
        self.supAlpha=[]

        if kernelopt[0]==0:
            self.kernel=lambda x1,x2 : self._kernel(x1,x2,0)
        elif kernelopt[0]==1:
            self.sigma=kernelopt[1]  # rbf kernal sigma
            self.kernel=lambda x1,x2 : self._kernel(x1,x2,1)


    def _kernel(self,x1,x2,type=1):
        n=len(x1) -1
        s=0

        if type==0 : #linear
            for i in range(n):
                s+= x1[i]*x2[i]
            return s

        if type==1: #rbf
            for i in range(n):
                s += (x1[i]-x2[i])**2
            k=math.exp(-s/(2 * self.sigma**2))
            return k

    def _select_i(self,dataset):
        m=len(dataset)
        for i in range(m):
            if  self.alpha[i]>0 and self.alpha[i] < self.C:
                p=self._predicate(dataset[i],dataset) * dataset[i][-1]
                if p!=1 :
                    return i

        for i in range(m):
            if self.alpha[i] == 0 :
                p=self._predicate(dataset[i],dataset) * dataset[i][-1]
                if p < 1:
                    return i
            elif self.alpha[i]==self.C:
                p=self._predicate(dataset[i],dataset)*dataset[i][-1]
                if p>1 :
                    return i
        return -1

    def _select_j(self,i,m):
        j=i
        while j==i:
            j=random.randint(0,m-1)
        return j

    def _update_alpha(self,i,j,dataset):
        low=0
        high=self.C
        if dataset[i][-1]==dataset[j][-1]:
            low=max(0,self.alpha[i]+self.alpha[j]-self.C)
            high=min(self.C,self.alpha[i]+self.alpha[j])
        else :
            low=max(0,self.alpha[j]-self.alpha[i])
            high=min(self.C,self.alpha[j]-self.alpha[i]+self.C)

        if low==high:
            return False

        rowi = dataset[i]
        rowj = dataset[j]

        eta= self.kernel(rowi,rowi) + self.kernel(rowj,rowj) - 2 * self.kernel(rowi,rowj)

        if eta == 0 :
            return False

        ei=self._predicate(rowi,dataset)-dataset[i][-1]#self._predicate(rowi,dataset) = f(x)
        ej = self._predicate(rowj,dataset)-dataset[j][-1]
        alpha_j = self.alpha[j]+dataset[j][-1]*(ei-ej)/eta
        if alpha_j==self.alpha[j]:
            return False
        if alpha_j > high:
            alpha_j = high
        if alpha_j < low:
            alpha_j = low


        Iold = self.alpha[i]
        self.alpha[i] += (self.alpha[j]-alpha_j) * dataset[i][-1]*dataset[j][-1]

        self.alpha[j]=alpha_j

        return True

    def _update_b(self,i,j,dataset):
        bi=self.b + dataset[i][-1] - self._predicate(dataset[i],dataset)
        bj=self.b + dataset[j][-1] - self._predicate(dataset[j],dataset)
        if self.alpha[i]> 0 and self.alpha[i] < self.C:
            return bi
        elif self.alpha[j] > 0 and self.alpha[j] < self.C :
            return bj
        return (bi+bi)/2.0

    def _predicate(self,row,dataset):
        m=len(dataset)
        y=0.0

        for i in range(m):
            y += self.alpha[i]*dataset[i][-1] * self.kernel(row,dataset[i])
        y+=self.b

        return y

    def smo(self,dataset):
        m=len(dataset)
        self.alpha=[0] * m

        for step in range(self.maxIter):
            changes=0

            i= self._select_i(dataset)
            if i== -1:
                break

            j=self._select_j(i,m)

            if not self._update_alpha(i,j,dataset):
                changes+=1
                continue
            self.b=self._update_b(i,j,dataset)

            if changes>100:
                break

        for idx, alpha in enumerate(self.alpha):
            self.W += np.multiply(alpha * dataset[idx][-1], dataset[idx])

        return self.W, self.b

if __name__ == '__main__':
    dataLoad = DataLoad('SVM/data01.txt')#watermelon20a.txt
    data = dataLoad.getFullData()
    realData = data.values[:, :].tolist()

    smo = SMO(2,[0])
    w,b = smo.smo(realData)
    print(w, b)
    class1 = []
    class2 = []

    for vector in realData:
        if vector[-1] == -1.0:
            class1.append(vector)
        else:
            class2.append(vector)

    plt.figure(1, facecolor='white')
    plt.scatter([x[0] for x in class1], [x[1] for x in class1], c='b')
    plt.scatter([x[0] for x in class2], [x[1] for x in class2], c='r')

    fig = plt.subplot()

    lineX1 = np.linspace(-2,10, 15)

    y1 = (-w[0] * lineX1 - b) / w[1]

    y2 = (-w[0] * lineX1 - b - 1) / w[1] #支撑平面
    y3 = (-w[0] * lineX1 - b + 1) / w[1] #支撑平面
    fig.plot(lineX1, y1, 'r--', lineX1, y2, 'g--', lineX1, y3, 'b--')

    plt.show()