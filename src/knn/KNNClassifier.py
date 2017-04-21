import sys
sys.path.append("..")
import matplotlib.pyplot as plt

from common.Distance import Distance
from common.DataLoad import DataLoad

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

    def predicate(self,item):
        nearestk=self._findNearestNeighbor(item)
        return nearestk[0]

if __name__ == '__main__':
    dataload = DataLoad('watermelon30a.txt')
    data = dataload.getRawData()
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
    print(knn.predicate([0.24, 0.39]))
    plt.scatter([0.24], [0.39], c='g')
    print(knn.predicate([0.53, 0.28]))
    plt.scatter([0.53], [0.28], c='y')
    plt.xlabel(u'密度', fontproperties='SimHei')
    plt.ylabel(u'含糖率', fontproperties='SimHei')
    plt.show()