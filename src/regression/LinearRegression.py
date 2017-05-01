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
        theta0_guess = 1.
        theta1_guess = 1.
        theta0_last = 100.
        theta1_last = 100.

        m = len(self.data)

        while (abs(theta1_guess-theta1_last) > variance or abs(theta0_guess - theta0_last) > variance):

            theta1_last = theta1_guess
            theta0_last = theta0_guess

            hypothesis = self.create_hypothesis(theta1_guess, theta0_guess)

            theta0_guess = theta0_guess - learning_rate * (1./m) * sum([ hypothesis(point[0]) - point[1] for point in self.data])
            theta1_guess = theta1_guess - learning_rate * (1./m) * sum([ (hypothesis(point[0]) - point[1]) * point[0] for point in self.data])

        return ( theta0_guess,theta1_guess )

    def create_hypothesis(self,theta1, theta0):
        return lambda x: theta1*x + theta0

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

    order = 9
    data = []
    for i in range(len(xa)):
        data.append([xa[i], ya[i]])

    alg = LinearRegressionAlg(order, data)
    matAA = alg.LeastSquareMethod()

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
    ax.plot(xxa,yya,color='g',linestyle='-',marker='')

    ax.legend()
    plt.show()