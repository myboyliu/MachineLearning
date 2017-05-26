from numpy import *
import pandas as pd

class GaussNavieBayes:
    def __init__(self, data, alpha):
        self.data = data
        self.alpha = alpha
        self.dim = 0
        self.summaryDict = {}
        self.priorP = {}
        self.MS = {} # 均值与方差
        self.posteriorP = {}

    def adjustRecord(self):
        for instance in self.data:
            vector = instance
            self.dim = len(vector) - 1
            if vector[-1] not in self.summaryDict:
                self.summaryDict[vector[-1]] = []
            self.summaryDict[vector[-1]].append([float(x) for x in instance[:-1]])
        print(self.summaryDict)

    def calcuteP(self):
        index = 1
        for key, value in self.summaryDict.items():
            if key not in self.priorP:
                self.priorP[key] = (len(value) + self.alpha) / (len(self.data) + self.dim * self.alpha)
            if key not in self.MS:
                self.MS[key] = []
                for index in range(self.dim):
                    l = [x[index] for x in value]
                    self.MS[key].append({})
                    self.MS[key][index] = [self.mean(l), self.stdev(l)]

        print(self.MS)

    def mean(self,numbers):
        return sum(numbers)/float(len(numbers))

    def stdev(self,numbers):
        avg = mean(numbers)
        variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
        return math.sqrt(variance)

    def trainNB(self):
        self.adjustRecord()
        self.calcuteP()
        print(self.priorP)
        print(self.posteriorP)

    def predicate(self, item):
        P = {}
        for key, value in self.MS.items():
            v = self.priorP[key]
            for index in range(len(item)):
                c = self.MS[key][index]
                v *= self.calculateProbability(item[index], c[0], c[1])
            P[key] = v
        print(P)
        return max(zip(P.values(),P.keys()))[1]

    def calculateProbability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

if __name__ == '__main__':
    data = pd.read_csv("xingbie.txt")
    rawData = data.values[:, 1:].tolist()

    GNB = GaussNavieBayes(rawData, 0.0)
    GNB.trainNB()
    print(GNB.predicate([6,130, 8]))