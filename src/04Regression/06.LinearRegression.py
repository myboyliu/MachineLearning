import matplotlib.pyplot as plt
import numpy as np
import math
import random

class LinearRegressionAlg(object):
    def __init__(self, order, data):
        self.order = order
        self.data = data

    def LeastSquareMethod(self):
        xa = [x[0] for x in self.data]
        ya = [x[1] for x in self.data]
        matA=[]
        for i in range(0,self.order+1):
            matA1=[]
            for j in range(0,self.order+1):
                tx=0.0
                for k in range(0,len(xa)):
                    dx=1.0
                    for l in range(0,j+i):
                        dx=dx*xa[k]
                    tx+=dx
                matA1.append(tx)
            matA.append(matA1)

        matA=np.array(matA)

        matB=[]
        for i in range(0,self.order+1):
            ty=0.0
            for k in range(0,len(xa)):
                dy=1.0
                for l in range(0,i):
                    dy=dy*xa[k]
                ty+=ya[k]*dy
            matB.append(ty)

        matB=np.array(matB)

        matAA=np.linalg.solve(matA,matB)

        return matAA

    def GradientDescentMethod(self, variance, learning_rate):
        matAA_guess = [1] * (self.order + 1)
        matAA_last = [100] * (self.order + 1)

        m = len(self.data)

        while (self.gradientDescentCheck(matAA_guess, matAA_last, variance) == True):
            for index in range(len(matAA_guess)):
                matAA_last[index] = matAA_guess[index]

            for index in range(len(matAA_guess)):
                hypothesis = self.create_hypothesis(matAA_guess, index)
                matAA_guess[index] = matAA_guess[index] - learning_rate * (1./m) * hypothesis

        return matAA_guess

    def create_hypothesis(self,matAA_guess, index):
        re = 0.0
        for j in range(len(self.data)):
            data = self.data[j]
            s = 0.0

            for i in range(len(matAA_guess)):
                s += matAA_guess[i] * (data[0]**i)

            re += (s - data[1]) * (data[0]**index)

        return re

    def gradientDescentCheck(self, matAA_guess, matAA_last, variance):
        re = False
        for i in range(len(matAA_guess)):
            value = abs(matAA_guess[i] - matAA_last[i])
            if (value > variance):
                re = True
                break
        return re

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(-1,1,0.02)
    y = [((a*a-1)*(a*a-1)*(a*a-1)+0.5)*np.sin(a*2) for a in x]

    i=0
    xa=[]
    ya=[]
    for xx in x:
        yy=y[i]
        d=float(random.randint(60,140))/100
        i+=1
        xa.append(xx*d)
        ya.append(yy*d)

    ax.plot(xa,ya,color='m',linestyle='',marker='.')

    order = 6
    data = []
    for i in range(len(xa)):
        data.append([xa[i], ya[i]])

    alg = LinearRegressionAlg(order, data)
    matAA = alg.LeastSquareMethod()
    print(matAA)
    matAA1 = alg.GradientDescentMethod(0.00001, 0.001)
    print(matAA1)
    xxa= np.arange(-1,1.06, 0.01)
    yya=[]
    for i in range(0,len(xxa)):
        yy=0.0
        for j in range(0,order+1):
            dy=1.0
            for k in range(0,j):
                dy*=xxa[i]
            dy*=matAA[j]
            yy+=dy
        yya.append(yy)

    yya1=[]
    for i in range(0,len(xxa)):
        yy=0.0
        for j in range(0,order+1):
            dy=1.0
            for k in range(0,j):
                dy*=xxa[i]
            dy*=matAA1[j]
            yy+=dy
        yya1.append(yy)

    ax.plot(xxa,yya,'g--', xxa, yya1,'r--')

    ax.legend()
    plt.show()