import numpy as np
import pandas as pd

class GarbageEmailClassifier():
    def __init__(self, data):
        self.data = data

    def createVocabList(self):
        vocabSet = set([])
        for document in self.data:
            print(document)
            vocabSet = vocabSet | set(document)

        return list(vocabSet)

    def seteOfWords2Vec(self, vocabList, inputSet):
        returnVec = [0] * len(vocabList)
        for word in inputSet:
            if word in vocabList:
                returnVec[vocabList.index(word)] = 1
            else:
                print("the word : ", word ," is not in my Vocabulary!")
        return returnVec

    def trainNB0(self, trainMatrix, trainCategory):
        numTrainDocs = len(trainMatrix)
        numWords = len(trainMatrix[0])
        pAbusive = sum(trainCategory) / float(numTrainDocs)
        p0Num = np.ones(numWords)
        p1Num = np.ones(numWords)
        p0Denom = 2.0
        p1Denom = 2.0
        for i in range(numTrainDocs):
            if trainCategory[i] == 1:
                p1Num += trainMatrix[i]
                p1Denom += sum(trainMatrix[i])
            else:
                p0Num += trainMatrix[i]
                p0Denom += sum(trainMatrix[i])

        p1Vect = p1Num / p1Denom
        p0Vect = p0Num / p0Denom
        return p0Vect, p1Vect, pAbusive

    def classifyNB(self, vec2Classify, p0Vec, p1Vec, pClass1):
        p1 = sum(vec2Classify * p1Vec) + pClass1
        p0 = sum(vec2Classify * p0Vec) + 1.0 - pClass1

        if p1 > p0:
            return 1
        else:
            return 0

if __name__ == '__main__':
    data = []
    labels = []

    with open("ge.txt") as ifile:
        for line in ifile:
            tokens = line.strip().split(',')
            data.append(tokens[:-1])
            labels.append(tokens[-1])

    ge = GarbageEmailClassifier(data)
    myVocabList = ge.createVocabList() # 全单词表
    print(myVocabList)
    trainMat = []
    for postingDoc in data:
        trainMat.append(ge.seteOfWords2Vec(myVocabList, postingDoc))
    print(trainMat)
    p0V, p1V, pAb = ge.trainNB0(np.array(trainMat), [float(x) for x in labels])

    testEntry = ['love', 'my', 'dalmation']

    thisDoc = np.array(ge.seteOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', ge.classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(ge.seteOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', ge.classifyNB(thisDoc, p0V, p1V, pAb))