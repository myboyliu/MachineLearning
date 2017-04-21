import matplotlib.pyplot as plt
import numpy as np
import math

# 最小二乘法
def linefit( x,y):
    N = len(x)
    sx,sy,sxx,syy,sxy=0,0,0,0,0
    for i in range(0,N):
        sx  += x[i]
        sy  += y[i]
        sxx += x[i]*x[i]
        syy += y[i]*y[i]
        sxy += x[i]*y[i]
    a = (sy*sx/N -sxy)/( sx*sx/N -sxx)
    b = (sy - a*sx)/N
    r = abs(sy*sx/N-sxy)/math.sqrt( (sxx-sx*sx/N)*(syy-sy*sy/N))
    return a,b,r

def create_hypothesis(theta1, theta0):
    return lambda x: theta1*x + theta0

# 梯度下降法
def linear_regression(data, learning_rate=0.001, variance=0.00001):
    theta0_guess = 1.
    theta1_guess = 1.
    theta0_last = 100.
    theta1_last = 100.

    m = len(data)

    while (abs(theta1_guess-theta1_last) > variance or abs(theta0_guess - theta0_last) > variance):

        theta1_last = theta1_guess
        theta0_last = theta0_guess

        hypothesis = create_hypothesis(theta1_guess, theta0_guess)

        theta0_guess = theta0_guess - learning_rate * (1./m) * sum([ hypothesis(point[0]) - point[1] for point in data])
        theta1_guess = theta1_guess - learning_rate * (1./m) * sum([ (hypothesis(point[0]) - point[1]) * point[0] for point in data])

    return ( theta0_guess,theta1_guess )

if __name__ == '__main__':
    x = np.arange(-2,2,0.1)
    y = 2*x+np.random.random(len(x))
    x = x.reshape((len(x),1))
    y = y.reshape((len(x),1))

    plt.figure(1, facecolor='white')
    plt.scatter(x, y)

    points = []
    for i in range(len(x)):
        points.append((float(x[i][0]), float(y[i][0])))

    lineX = np.linspace(-2,2, 15)
    fig = plt.subplot()

    b, a = linear_regression(points, 0.001, 0.00001)
    y1 = a * lineX + b

    m, t,r = linefit(x, y)
    y2 = m * lineX + t

    fig.plot(lineX, y1, 'r--', lineX, y2, 'g--')

    plt.show()