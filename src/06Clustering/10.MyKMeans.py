from numpy import *
import sys
import os

import shutil
sys.path.append("..")
from common.DataLoad import DataLoad
from common.Distance import Distance

class KMeans():
    def __init__(self,data, K, IsKMeansPlusPlus = True):
        self._data = data
        self._normalizeData = [] #样本数据
        self._comment = [] #类别数据
        self._K = K
        self._clusterCenters = [] #K个中心点的数据
        self._memberOfClusters = [] #每个样本所属中心点的数据
        self._isKMeansPlusPlus = IsKMeansPlusPlus
        self.transformData()

    def transformData(self): # 初始化_normalizeData, _comment, _memberOfClusters
        lineIdx = 0
        for record in self._data:

            vector = []
            for cid, cell in enumerate(record):
                if cid == 0:
                    self._comment.append(cell)
                else:
                    vector.append(cell)
            self._normalizeData.append(vector)
            self._memberOfClusters.append(lineIdx)
            lineIdx += 1

    def kCluster(self):
        changed = True
        self.initCenters()

        while changed:
            pointChangedNumber = self.assignPointToCluster()
            if pointChangedNumber == 0:
                changed = False
            else:
                self.updateCenters()
                changed = True
        return self._memberOfClusters, self._clusterCenters

    def initCenters(self):
        self._clusterCenters = []
        firstCenter = random.choice(range(len(self._normalizeData))) #从样本节点中随机选取一个节点作为第一个中心点
        self._clusterCenters.append(self._normalizeData[firstCenter])

        for i in range(self._K - 1):
            weights = [self.selectClosedInstanceToCenters(x) for x in self._normalizeData]
            total = sum(weights)
            weights = [(x/total) for x in weights]
            maxNum = random.random()
            index = -1
            total = 0.0
            while total < maxNum:
                index += 1
                total += weights[index]

            self._clusterCenters.append(self._normalizeData[index])
    def selectClosedInstanceToCenters(self, v):
        result = Distance.computeEuDistance(v, self._clusterCenters[0])

        for vector in self._clusterCenters[1:]:
            distance = Distance.computeEuDistance(v, vector)
            if distance < result:
                result = distance

        return result
    def assignPointToCluster(self):
        pointChangedNo = 0
        for idx, record in enumerate(self._normalizeData):
            clusterIdx = self.selectClosestCenterOfCluster(record)
            if clusterIdx != self._memberOfClusters[idx]:
                pointChangedNo += 1
                self._memberOfClusters[idx] = clusterIdx
        return pointChangedNo
    def selectClosestCenterOfCluster(self, v): #找到离节点v最近的中心，返回这个中心在_clusterCenter中的index，也就是这个中心所在的簇号
        mindistance = 9999999
        clusterIdx = -1
        for centerIndex, center in enumerate(self._clusterCenters):
            distance = Distance.computeEuDistance(v, center)
            if distance < mindistance:
                clusterIdx = centerIndex
                mindistance = distance

        return clusterIdx
    def updateCenters(self): #计算每个簇中所有节点的中心点，其实就是每个簇中所有节点对应的每一个向量维度的均值
        clusterMemberCnt = [0] * self._K
        newClusterCenters = [[0] * len(self._normalizeData[0]) for i in range(self._K)]
        for idx, point in enumerate(self._normalizeData):
            clusterMemberCnt[self._memberOfClusters[idx]] += 1
            for colIdx, cell in enumerate(point):
                newClusterCenters[self._memberOfClusters[idx]][colIdx] += float(cell)

        for i in range(self._K):
            for j in range(len(newClusterCenters[0])):
                self._clusterCenters[i][j] = newClusterCenters[i][j] * 1.0 / clusterMemberCnt[i]

if __name__ == '__main__':
    dataLoad = DataLoad('06Clustering.txt')
    K = 5
    data = dataLoad.getFullData()
    mCluster, centers = km.kCluster()
    print(mCluster)
    basedir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dir = os.path.join(basedir, "images/06Clustering/cluster/")

    clustersDir = os.path.join(dir, "Clusters/")
    shutil.rmtree(clustersDir)
    os.mkdir(clustersDir)

    for i in range(K):
        clusterDir=os.path.join(clustersDir,'cluster-{}'.format(i))
        if not os.path.isdir(clusterDir):
            os.mkdir(clusterDir)

    for idx,clusterid in enumerate(mCluster):
        src=km._comment[idx]
        filename=os.path.basename(src)
        dest = os.path.join(os.path.join(clustersDir,'cluster-{}'.format(clusterid)),filename)
        shutil.copyfile(src,dest)