import numpy as np
import TreePlotter as tp
import copy
import re
from math import log
import pandas as pd

class DecisionTree():
    """
    compareType: 0 : 信息增益， 1：信息增益率，2，基尼系数
    prunType : 0 : 不剪枝，1：预剪枝，2：后剪枝
    """
    def __init__(self, dataAll, columnsAll, compareType = 0, prunType = 0):
        self.dataAll = copy.deepcopy(dataAll)
        self.columnsAll = copy.deepcopy(columnsAll)

        self.alType = compareType
        self.prunType = prunType

    def createTree(self,dataSet, dataColumns, dataTest):
        mytree = self.buildTree(dataSet, dataColumns, dataTest)
        #precisions = self.getTreePrecision(mytree, dataTest, copy.deepcopy( self.columnsAll ))
        if self.prunType == 2 or self.prunType == 3:
            mytree = self.postPruningTree(mytree, dataSet, dataTest, copy.deepcopy( self.columnsAll ))
            precisions = self.getTreePrecision(mytree, dataTest,copy.deepcopy( self.columnsAll ))
            print(precisions)
        return mytree
    def postPruningTree(self, inputTree,dataSet,data_test,labels):
        firstStr=list(inputTree.keys())[0]
        secondDict=inputTree[firstStr]
        classList=[example[-1] for example in dataSet]
        featkey=copy.deepcopy(firstStr)
        if '<=' in firstStr:
            featkey=re.compile("(.+<=)").search(firstStr).group()[:-2]
            featvalue=float(re.compile("(<=.+)").search(firstStr).group()[2:])
        labelIndex=labels.index(featkey)
        temp_labels=copy.deepcopy(labels)
        del(labels[labelIndex])
        for key in secondDict.keys():
            if type(secondDict[key]).__name__=='dict':
                if type(dataSet[0][labelIndex]).__name__=='str':
                    inputTree[firstStr][key]=\
                        self.postPruningTree(secondDict[key], self.splitData(dataSet,labelIndex,key),self.splitData(data_test,labelIndex,key),copy.deepcopy(labels))
                else:
                    inputTree[firstStr][key]=self.postPruningTree(secondDict[key],
                                                             self.splitContinuousDataSet(dataSet,labelIndex,featvalue,key), \
                                                             self.splitContinuousDataSet(data_test,labelIndex,featvalue,key), \
                                                             copy.deepcopy(labels))
        precisionP = self.testingMajor(self.majorityCnt(classList),data_test)
        precisionN = self.testing(inputTree,data_test,temp_labels)
        if  precisionN >= precisionP :
            return inputTree
        return self.majorityCnt(classList)
    def buildTree(self, dataSet, dataColumns, dataTest, precision = 0.0, priWeight = []):
        '''
        1. 首先找到信息增益最大的那个特征的index
        2. 根据index找到特征的label，然后把他放到当前树的根节点
        3. 找到这个特征下面的所有值. 根据这个值来拆分数据，然后分别开始递归
        '''
        if priWeight == [] :
            priWeight = [1] * len(dataSet)

        classList=[example[-1] for example in dataSet]
        if classList.count(classList[0])==len(classList):
            return classList[0]
        if len(dataSet[0])==1:
            return self.majorityCnt(classList)

        temp_labels=copy.deepcopy(dataColumns)
        bestFeatureIndex = self.getBestFeatureToSplit(dataSet, dataColumns,[], priWeight)
        bestFeatureLabel = dataColumns[bestFeatureIndex]

        if precision == 0.0:
            precision = self.testingMajor(self.majorityCnt(classList),dataTest)
        data1 = precision
        if self.prunType == 0: # 不剪枝
            mytree = {bestFeatureLabel:{}}
        elif self.prunType == 1: # 预剪枝
            data1 = self.testing_feat(bestFeatureLabel,dataSet,dataTest,temp_labels)# 划分后
            #data2 = self.testingMajor(self.majorityCnt(classList),dataTest) # 划分前
            if precision < data1:
                mytree = {bestFeatureLabel:{}}
            else:
                return self.majorityCnt(classList)
        elif self.prunType == 2: #悲观剪枝法
            mytree = {bestFeatureLabel:{}}
        elif self.prunType == 3: #代价复杂度剪枝法
            mytree = {bestFeatureLabel:{}}

        featureValues = [x[bestFeatureIndex] for x in dataSet]
        uniqeFeatureValues = set(featureValues)
        if '-' in uniqeFeatureValues:
            uniqeFeatureValues.remove('-')

        if type(dataSet[0][bestFeatureIndex]).__name__=='str':
            currentlabel=self.columnsAll.index(dataColumns[bestFeatureIndex])
            featValuesFull=[example[currentlabel] for example in self.dataAll]
            uniqueValsFull=set(featValuesFull)
            if '-' in uniqueValsFull:
                uniqueValsFull.remove('-')

        lenTotal = len(self.getRealData(bestFeatureIndex, dataSet))

        del (dataColumns[bestFeatureIndex])
        for uniqe in uniqeFeatureValues:
            subdata, preWeight, noValueCount = self.splitDataWithWeight(dataSet, bestFeatureIndex, uniqe, 1)
            for i in range(len(preWeight)):
                if preWeight[i] == '-':
                    preWeight[i] = (len(subdata) - noValueCount) / lenTotal
            subLabels=dataColumns[:]
            if type(dataSet[0][bestFeatureIndex]).__name__=='str':
                uniqueValsFull.remove(uniqe)
            mytree[bestFeatureLabel][uniqe] = self.buildTree(subdata, subLabels, dataTest, data1, preWeight)

        if type(dataSet[0][bestFeatureIndex]).__name__=='str':
            for value in uniqueValsFull:
                mytree[bestFeatureLabel][value]=self.majorityCnt(classList)

        return mytree
    def getBestFeatureToSplit(self, dataSet, dataColumns, dataTest = [], priWeight = []):
        '''
        首先计算出信息熵
        然后循环所有特征，计算出每个特征条件熵，计算出信息增益，找到信息增益最大的那个特征的index，返回，但是具体的对于离散型
        和连续型，会有不同
        '''
        numFeatures = len(dataSet[0]) - 1 # 特征的数量，-1表示去掉最后一列的类别列
        bestFeature = -1

        bestInfoGain = 0.0 if self.alType == 0 or self.alType == 1 else 10000.0
        bestSplitDict={}
        for i in range(numFeatures):
            baseEntropy = self.getValueOfCompare(i, dataSet, priWeight)
            realData = self.getRealData(i, dataSet)
            featureColumnData = [x[i] for x in dataSet]
            # 区分连续型和离散型特征
            if type(featureColumnData[0]).__name__ == 'float' or type(featureColumnData[0]).__name__ == 'int':
                #产生n-1个候选划分点
                sortfeatList=sorted(featureColumnData)
                splitList=[]
                for j in range(len(sortfeatList)-1):
                    splitList.append(round((sortfeatList[j]+sortfeatList[j+1])/2.0, 4))

                bestSplitEntropy=10000
                slen=len(splitList)
                #求用第j个候选划分点划分时，得到的信息熵，并记录最佳划分点
                for j in range(slen):
                    value=splitList[j]
                    newEntropy=0.0
                    subDataSet0=self.splitContinuousDataSet(dataSet,i,value,0) #大于当前均值的样本列表
                    subDataSet1=self.splitContinuousDataSet(dataSet,i,value,1) #小于当前均值的样本列表
                    prob0=len(subDataSet0)/float(len(dataSet))
                    newEntropy+=prob0*self.getValueOfCompare(i, subDataSet0, priWeight)
                    prob1=len(subDataSet1)/float(len(dataSet))
                    newEntropy+=prob1*self.getValueOfCompare(i, subDataSet1, priWeight)
                    if newEntropy<bestSplitEntropy:
                        bestSplitEntropy=newEntropy
                        bestSplit=j
                        #用字典记录当前特征的最佳划分点
                bestSplitDict[dataColumns[i]]=splitList[bestSplit]
                # newEntry为当前特征的信息熵，也就是条件熵，为当前特征的所有可能取值的条件熵的和
                # infoGain就是信息增益
                if self.alType == 0: #比较信息增益，找到信息增益最大的
                    infoGain = baseEntropy - bestSplitEntropy
                elif self.alType == 1: #比较信息增益率，找到信息增益率最大的
                    infoGain = (baseEntropy - bestSplitEntropy) / baseEntropy
                elif self.alType == 2: #比较基尼系数，找到基尼系数最小的
                    infoGain = bestSplitEntropy
                #对离散型特征进行处理
            else:
                # 过滤掉当前特征的重复值，然后根据每一个取值进行分割样本，计算出相关熵，然后相加得到条件熵
                uniqeVec = set(featureColumnData)
                if '-' in uniqeVec:
                    uniqeVec.remove('-')

                newEntry = 0.0

                for value in uniqeVec:
                    #取出dataset中第i列的内容，然后过滤出等于value的所有样本，当然结果中会去掉i这一列的
                    splittedVec = self.splitData(dataSet, i, value, 0)

                    prob = len(splittedVec) / len(realData)
                    newEntry += prob * self.getValueOfCompare(i, splittedVec, priWeight)

                # newEntry为当前特征的信息熵，也就是条件熵，为当前特征的所有可能取值的条件熵的和
                # infoGain就是信息增益
                if self.alType == 0: #比较信息增益，找到信息增益最大的
                    infoGain = baseEntropy - newEntry
                elif self.alType == 1: #比较信息增益率，找到信息增益率最大的
                    infoGain = (baseEntropy - newEntry) / baseEntropy
                elif self.alType == 2: #比较基尼系数，找到基尼系数最小的
                    infoGain = newEntry

            p = len(realData) / len(dataSet)
            infoGain = p * infoGain
            if self.alType == 0 or self.alType == 1:
                if infoGain > bestInfoGain:
                    bestInfoGain = infoGain
                    bestFeature = i
            else:
                if infoGain < bestInfoGain:
                    bestInfoGain = infoGain
                    bestFeature = i
        if type(dataSet[0][bestFeature]).__name__=='float' or type(dataSet[0][bestFeature]).__name__=='int':
            bestSplitValue=bestSplitDict[dataColumns[bestFeature]]
            dataColumns[bestFeature]=dataColumns[bestFeature]+'<='+str(bestSplitValue)
            for i in range((np.array(dataSet)).shape[0]):
                if dataSet[i][bestFeature]<=bestSplitValue:
                    dataSet[i][bestFeature]=1
                else:
                    dataSet[i][bestFeature]=0

        return bestFeature
    def getRealData(self, index, dataSet):
        re = []
        for vector in dataSet:
            if vector[index] == '-':
                pass
            else:
                re.append(vector)
        return re
    def getValueOfCompare(self, index, dataSet, priWeight):
        if self.alType == 0 or self.alType == 1:
            return self.getEntropyOfDataSet(index, dataSet, priWeight)
        elif self.alType == 2:
            return self.getGiniOfDataSet(dataSet, priWeight)
        else:
            return 0
    ### 得到某个特征的某一个值的信息熵
    def getEntropyOfDataSet(self, index,  dataSet, priWeight):
        num = 0
        labels = {}
        for idx, vector in enumerate(dataSet):
            if vector[index] != '-':
                currentLabel = vector[-1]
                if currentLabel not in labels.keys():
                    labels[currentLabel] = 0
                labels[currentLabel] += 1 * priWeight[idx]
                num += 1 * priWeight[idx]

        entropyValue = 0.0
        for key in labels:
            prop = float(labels[key]) / num
            entropyValue -= prop * log(prop, 2)

        return entropyValue
    ### 得到某个特征的某一个值的基尼系数
    def getGiniOfDataSet(self, dataSet, priWeight):
        numEntries=0
        labelCounts={}
        #给所有可能分类创建字典
        for idx, featVec in enumerate(dataSet):
            currentLabel=featVec[-1]
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel]=0
            labelCounts[currentLabel] += 1 * priWeight[idx]
            numEntries += 1 * priWeight[idx]
        Gini=1.0
        #以2为底数计算香农熵
        for key in labelCounts:
            prob = float(labelCounts[key])/numEntries
            Gini-=prob*prob
        return Gini
    def splitData(self, dataSet, index, value, IsRemoveCurrentColumn):
        result = []
        for vector in dataSet:
            re = []
            if IsRemoveCurrentColumn == 1:
                if vector[index] == value or vector[index] == '-':
                    re = vector[:index]
                    re.extend(vector[index+1:])
                    result.append(re)
            else:
                if vector[index] == value:
                    result.append(vector)
        return result
    def splitDataWithWeight(self, dataSet, index, value, IsRemoveCurrentColumn):
        result = []
        preWeight = []
        noValueCount = 0
        for vector in dataSet:
            re = []
            if vector[index] == value or vector[index] == '-':
                re = vector[:index]
                re.extend(vector[index+1:])
                result.append(re)
                if vector[index] == value:
                    preWeight.append(1)
                else:
                    preWeight.append('-')
                    noValueCount += 1
        return result, preWeight, noValueCount
    def majorityCnt(self,classList):
        classCount={}
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote]=0
            classCount[vote]+=1
        return max(classCount)
    def splitContinuousDataSet(self,dataSet,axis,value,direction):
        retDataSet=[]
        for featVec in dataSet:
            if direction==0:
                if featVec[axis]>value:
                    reducedFeatVec=featVec[:axis]
                    reducedFeatVec.extend(featVec[axis+1:])
                    retDataSet.append(reducedFeatVec)
            else:
                if featVec[axis]<=value:
                    reducedFeatVec=featVec[:axis]
                    reducedFeatVec.extend(featVec[axis+1:])
                    retDataSet.append(reducedFeatVec)
        return retDataSet
    def classify(self, inputTree,featLabels,testVec):
        firstStr=list(inputTree.keys())[0]
        if '<=' in firstStr:
            featvalue=float(re.compile("(<=.+)").search(firstStr).group()[2:])
            featkey=re.compile("(.+<=)").search(firstStr).group()[:-2]
            secondDict=inputTree[firstStr]
            featIndex=featLabels.index(featkey)
            if testVec[featIndex]<=featvalue:
                judge=1
            else:
                judge=0
            for key in secondDict.keys():
                if judge==int(key):
                    if type(secondDict[key]).__name__=='dict':
                        classLabel=self.classify(secondDict[key],featLabels,testVec)
                    else:
                        classLabel=secondDict[key]
        else:
            secondDict=inputTree[firstStr]
            featIndex=featLabels.index(firstStr)
            for key in secondDict.keys():
                if testVec[featIndex]==key:
                    if type(secondDict[key]).__name__=='dict':
                        classLabel=self.classify(secondDict[key],featLabels,testVec)
                    else:
                        classLabel=secondDict[key]
        return classLabel
    def testing_feat(self, feat,train_data,test_data,labels):
        class_list = [example[-1] for example in train_data]
        bestFeatIndex = labels.index(feat)
        train_data = [example[bestFeatIndex] for example in train_data]
        test_data = [(example[bestFeatIndex],example[-1]) for example in test_data]
        all_feat = set(train_data)
        correct = 0.0
        for value in all_feat:
            class_feat = [ class_list[i] for i in range(len(class_list)) if train_data[i]==value]
            major = self.majorityCnt(class_feat)
            for data in test_data:
                if data[0]==value and data[1]==major:
                    correct+=1.0
        # print 'myTree %d' % error
        return correct / len(test_data)
    def testingMajor(self,major,data_test):
        if len(data_test) == 0:
            return 1
        correct = 0.0
        for i in range(len(data_test)):
            if major==data_test[i][-1]:
                correct+=1
        return float(correct) / len(data_test)
    def testing(self,myTree, data_test, labels):
        correct = 0.0
        for i in range(len(data_test)):
            if self.classify(myTree, labels, data_test[i]) == data_test[i][-1]:
                correct += 1
        # print 'myTree %d' % error
        return float(correct) / len(data_test)
    def getTreePrecision(self, myTree, dataTest, dataColumns):
        correct = self.getTotalCorrectCase(myTree, dataTest, dataColumns)
        return correct / len(dataTest)
    def getTotalCorrectCase(self, myTree, dataTest, dataColumns):
        firstStr=list(myTree.keys())[0]
        secondDict=myTree[firstStr]
        labelIndex=dataColumns.index(firstStr)
        s = 0
        for key in secondDict.keys():
            data = self.splitData(dataTest, labelIndex, key)
            if type(secondDict[key]).__name__=='dict':
                s += self.getTotalCorrectCase(secondDict[key], data, dataColumns)
            else:
                major=secondDict[key]
                if data == []:
                    a = 0
                else:
                    a = self.testingMajor(major, data) * len(data)

                s += a
        return s
if __name__ == '__main__':
    # dataLoad = DataLoad('watermelon20.txt')
    # data = dataLoad.getFullData()
    # readData = data.values[:11, 1:].tolist()
    # columns = data.columns[1:-1].tolist()
    # dataTest = data.values[11:,1:].tolist()

    dataLoad = pd.read_csv('watermelon20a.txt')
    readData = dataLoad.values[:, 1:].tolist()
    columns = dataLoad.columns[1:-1].tolist()
    dataTest = []

    dt = DecisionTree(readData, columns,0, 0)
    mytree = dt.createTree(readData, columns, dataTest)
    tp.createPlot(mytree)
    print(mytree)