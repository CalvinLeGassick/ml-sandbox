import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from naive_bayes import NBClassifier
from knn import KNNClassifier

trainFile = "train.txt"
testFile = "test.txt"
trainHeaderList = ["ID", "Refractive Index", "Sodium", "Magnesium", "Aluminum", "Silicon", "Potassium", "Calcium", "Barium", "Iron", "Classification"]
testHeaderList = ["Refractive Index", "Sodium", "Magnesium", "Aluminum", "Silicon", "Potassium", "Calcium", "Barium", "Iron", "Classification"]

def loadTrainingFile(filename, headers):
    return pd.pandas.read_csv(filename, names=headers).drop("ID", 1)

def loadTestFile(filename, headers):
    return pd.pandas.read_csv(filename, names=headers)

def measureAccuracy(df, classifier, setName):
    success = 0
    for index, row in df.iterrows():
        classification = classifier.classify(row[:-1])
        if classification == row["Classification"]:
            success += 1

    print "Accuracy on {0} with {1} classifier: {2}".format(setName, classifier.name, success/float(len(df)))

def measureLeaveOneOutAccuracyKNN(df, classifier, setName):
    success = 0
    for index, row in df.iterrows():
        classifier.setTrain(df.drop(index))
        classification = classifier.classify(row[:-1])
        if classification == row["Classification"]:
            success += 1

    print "Leave one out accuracy on {0} with {1} classifier: {2}".format(setName, classifier.name, success/float(len(df)))

def runNaiveBayesClassification(trainDF, testDF):
    nb = NBClassifier()
    nb.train(trainDF)

    print "\nRUNNING GAUSSIAN NAIVE BAYES CLASSIFIER"
    measureAccuracy(trainDF, nb, "Train")
    measureAccuracy(testDF, nb, "Test")
    print "#####################################"
    print

def runKnnClassification(trainDF, testDF):
    print "RUNNING KNN CLASSIFIER L1"
    knn = KNNClassifier()
    knn.setL(1)
    for x in [1, 3, 5, 7]:
        print "k = {0}".format(x)
        knn.setK(x)
        measureLeaveOneOutAccuracyKNN(trainDF, knn, "Training")
        knn.setTrain(trainDF)
        measureAccuracy(testDF, knn, "Testing")
        print
    print "#####################################"
    print "\n"

    print "RUNNING KNN CLASSIFIER L2"
    knn.setL(2)
    for x in [1, 3, 5, 7]:
        print "k = {0}".format(x)
        knn.setK(x)
        measureLeaveOneOutAccuracyKNN(trainDF, knn, "Training")
        knn.setTrain(trainDF)
        measureAccuracy(testDF, knn, "Testing")
        print
    print "#####################################"
    print "\n"

def main():
    trainDF = loadTrainingFile(trainFile, trainHeaderList)
    testDF = loadTestFile(testFile, testHeaderList)

    # Look at distribution of classes in training
    fig = plt.figure()
    trainDF["Classification"].hist()
    plt.savefig("class_histogram.png")

    # Train and evaluate NB classifier.
    runNaiveBayesClassification(trainDF, testDF)

    # Train and evaluatate
    runKnnClassification(trainDF, testDF)

if __name__ == "__main__":
   main()
