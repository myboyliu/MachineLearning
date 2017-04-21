from numpy import *
import sys
sys.path.append("..")
from common.DataLoad import DataLoad


class MultinomialNavieBayes():
    def __init__(self, data, alpha):
        self.data = data
        self.alpha = alpha
        self.dim = 0
        self.summaryDict = {}
        self.priorP = {}
        self.posteriorP = {}

    def adjustRecord(self):
        for instance in self.data:
            vector = instance
            self.dim = len(vector) - 1
            if vector[-1] not in self.summaryDict:
                self.summaryDict[vector[-1]] = []
            self.summaryDict[vector[-1]].append(vector)

    def calcuteP(self):
        index = 1
        for key, value in self.summaryDict.items():
            if key not in self.priorP:
                self.priorP[key] = (len(value) + self.alpha) / (len(self.data) + self.dim * self.alpha)

            if key not in self.posteriorP:
                self.posteriorP[key] = []

            for idx in range(self.dim):
                self.posteriorP[key].append({})

            for instance in value:
                for idx in range(len(instance) - 1):
                    if instance[idx] not in self.posteriorP[key][idx]:
                        self.posteriorP[key][idx][instance[idx]] = 1
                    else:
                        self.posteriorP[key][idx][instance[idx]] += 1

        for key, value in self.posteriorP.items():
            cnt = len(self.summaryDict[key])
            for instance in value:
                for k, v in instance.items():
                    instance[k] = (v + self.alpha) / (cnt + self.alpha * self.dim)

    def trainNB(self):
        self.adjustRecord()
        self.calcuteP()
        print(self.priorP)
        print(self.posteriorP)

    def predicate(self, item):
        P = {}
        for key, value in self.posteriorP.items():
            v = self.priorP[key]
            for index in range(len(item)):
                c = self.posteriorP[key][index]
                label = (str)(item[index])
                v *= (float)(c[label])
            P[key] = v
        print(P)
        return max(zip(P.values(),P.keys()))[1]

if __name__ == '__main__':
    dataLoad = DataLoad("weatherAndtennis.txt")
    data = dataLoad.getRawData()
    MNB = MultinomialNavieBayes(data, 1.0)
    MNB.trainNB()
    print(MNB.predicate(['Sunny', 'Cool', 'High', 'Strong']))