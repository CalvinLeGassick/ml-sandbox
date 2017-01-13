import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

class KNNClassifier:
    name = "Nearest Neighbor"
    classifierHeader = "Classification"

    classes = []
    attributes = []
    columnDivisors = {}
    columnMeans = {}

    def normalizeDF(self, df):
        classColumn = df[self.classifierHeader]

        # Get attributes of Training Data (Needed for normalizing future input)
        self.classes = classColumn.unique()
        self.attributes = [name for name in df.columns if name != self.classifierHeader]
        self.columnMeans = df[self.attributes].mean()
        self.columnDivisors = df[self.attributes].std()

        normalized = df.copy(deep=True)
        normalized[self.attributes] = (df[self.attributes] - self.columnMeans) / self.columnDivisors
        return normalized

    def normalizeSinglePoint(self, inputFrame):
        assert self.train is not None
        normalized = inputFrame.copy(True)
        for a in self.attributes:
            if self.columnDivisors[a] == 0:
                if inputFrame[a] == self.columnMeans[a]:
                    normalized[a] = 1
                else:
                    normalized[a] = 0
            else:
                normalized[a] = (normalized[a] - self.columnMeans[a])/self.columnDivisors[a]
        return normalized

    def norm(self, a, b, L):
        attrs = self.attributes
        return (abs(a - b[attrs]) ** L).sum() ** (1/float(L))

    def setK(self, k):
        self.k = k

    def setL(self, L):
        self.L = L

    def setTrain(self, train):
        self.train = self.normalizeDF(train)

    def classify(self, inputFrame):
        assert self.train is not None
        k = self.k
        L = self.L
        unknown = self.normalizeSinglePoint(inputFrame)
        knn = self.computeKNN(unknown, k, L)
        # print "K closest:"
        # for p in knn:
        #     print "{}: {}".format(p[self.classifierHeader], self.norm(unknown, p, L))

        classes = []
        lengthToClassMap = {}
        for p in knn:
            if p[self.classifierHeader] not in classes:
                classes.append(p[self.classifierHeader])
        for c in classes:
            l = [p for p in knn if p[self.classifierHeader] == c]
            if len(l) not in lengthToClassMap:
                lengthToClassMap[len(l)] = {}
            lengthToClassMap[len(l)][c] = l

        mostVoters = max(lengthToClassMap.keys())
        if len(lengthToClassMap[mostVoters]) > 1:
            potential = []
            for c in lengthToClassMap[mostVoters].keys():
                potential += lengthToClassMap[mostVoters][c]

            bestDist = self.norm(unknown, potential[0], L)
            bestClass = potential[0][self.classifierHeader]
            for p in potential:
                if self.norm(unknown, p, L) < bestDist:
                    bestDist = self.norm(unknown, p, L)
                    bestClass = p[self.classifierHeader]

            return bestClass
        else:
            return lengthToClassMap[mostVoters].keys()[0]

    def computeKNN(self, unknownPoint, k, L):
        inputPoints = self.train
        distances = []
        for index, point in inputPoints.iterrows():
            dist = self.norm(unknownPoint, point, L)
            distances.append((dist, point))
        distances.sort(key=lambda x:x[0])
        knn = [d[1] for d in distances[:k]]
        return knn
