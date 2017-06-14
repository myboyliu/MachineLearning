import matplotlib.pyplot as plt
import pandas as pd
from Distance import Distance

class KNNClassifier():
    def __init__(self,data, k=10):
        self.data = data
        self.k=k

    def _findNearestNeighbor(self,item):
        neighbors=[]

        for idx, onerecord in enumerate(self.data):
            distance=Distance.computeManhattanDistance(onerecord[:-1],item)
            neighbors.append([idx, onerecord[-1],distance])

        neighbors.sort(key=lambda x:x[2])

        return neighbors

    def predicate(self,item, k):
        nearestk=self._findNearestNeighbor(item)
        res = {'N':0,'Y':0}
        for vector in nearestk[:k]:
            if vector[1] == '否':
                res['N'] += 1
            else:
                res['Y'] += 1
        print(res)
        return nearestk[0][1]

if __name__ == '__main__':
    dataload = pd.read_csv('watermelon30a.txt')
    data = dataload.values[:, 1:].tolist()
    knn = KNNClassifier(data)

    dataForClassOne = []
    dataForClassTwo = []

    for vector in data:
        if vector[-1] == '是':
            dataForClassOne.append(vector)
        else:
            dataForClassTwo.append(vector)

    plt.figure(1, facecolor='white')
    plt.scatter([x[0] for x in dataForClassOne], [x[1] for x in dataForClassOne], c = 'r')
    plt.scatter([x[0] for x in dataForClassTwo], [x[1] for x in dataForClassTwo], c = 'k')
    K = 1
    print(knn.predicate([0.24, 0.39], 5))
    plt.scatter([0.24], [0.39], c='g')
    print(knn.predicate([0.53, 0.28], 5))
    plt.scatter([0.53], [0.28], c='y')
    plt.xlabel(u'密度', fontproperties='SimHei')
    plt.ylabel(u'含糖率', fontproperties='SimHei')
    plt.show()