import numpy
import sys
sys.path.append("..")
import random
import math

from common.DataLoad import DataLoad

import csv
import random
import math

def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    print(separated)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            v = calculateProbability(x, mean, stdev)
            probabilities[classValue] *= v
    return probabilities

def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def main():
    filename = 'pima-indians.txt'
    splitRatio = 0.67
    dataLoad = DataLoad(filename)

    dataset = dataLoad.getfilerecord()
    trainingSet, testSet = splitDataset(dataset, 0.9)
    print('---------')
    print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
    # prepare model
    summaries = summarizeByClass(trainingSet)
    print(testSet)
    print('---------')
    print(summaries)
    # test model
    print('---------')
    predictions = getPredictions(summaries, testSet)
    print(predictions)

    # accuracy = getAccuracy(testSet, predictions)
    # print('Accuracy: {0}%'.format(accuracy))

if __name__ == '__main__':
    #main()
    dataset = [[1,20,1], [2,21,1], [3,22,1], [4,22,0], [12, 24, 0]]
    summaries = summarizeByClass(dataset)
    print('Summary by class value: {0}'.format(summaries))
    inputVector = [[2.0, 21.0], [8.0, 21.5]]
   # result = predict(summaries, inputVector)
    # print('Prediction: {0}'.format(result))
    predictions = getPredictions(summaries, inputVector)
    print(predictions)
    accuracy = getAccuracy(inputVector, predictions)
    print('Accuracy: {0}'.format(accuracy))