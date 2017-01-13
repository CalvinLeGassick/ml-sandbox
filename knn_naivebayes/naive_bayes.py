import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

class NBClassifier:
    trained = False
    name = "Naive Bayes"
    classLikelihood = {}
    classToAttributeMeans = {}
    classToAttributeVariances = {}
    classes = []
    attributes = []

    classToNormalizers = {}

    classifierHeader = "Classification"

    def normalPDF(self, mean, variance, x):
        if variance == 0:
            return 0 if x != mean else 1
        exponent = math.exp(-(math.pow(x-mean,2)/(2*variance)))
        return exponent * (1 / (variance * math.sqrt(2*math.pi)))

    def summarizeInputData(self):
        for c in self.classes:
            print "Prior likelihood of class {0}: {1}".format(c, self.classLikelihood[c])

        for c in self.classes:
            for d in self.attributes:
                print "mean of attribute {0} given class {1}: {2}".format(d, c, self.classToAttributeMeans[c][d])

        for c in self.classes:
            for d in self.attributes:
                print "variance of attribute {0} given class {1}: {2}".format(d, c, self.classToAttributeVariances[c][d])

    def normalizeDF(self, df):
        self.columnMeans = df[self.attributes].mean()
        self.columnDivisors = df[self.attributes].std()

        normalized = df.copy(deep=True)
        normalized[self.attributes] = (df[self.attributes] - self.columnMeans) / self.columnDivisors
        return normalized

    def train(self, df):
        classColumn = df[self.classifierHeader]
        self.classes = classColumn.unique()
        self.attributes = [name for name in df.columns if name != self.classifierHeader]

        # Normalize Data
        df = self.normalizeDF(df)

        # Get mu_{jk} and sigma_{jk}
        dataPointsWithClassC = {}
        for c in self.classes:
            dataPointsWithClassC[c] = df.loc[df[self.classifierHeader] == c]
            self.classToAttributeMeans[c] = {}
            self.classToAttributeVariances[c] = {}
            self.classLikelihood[c] = len(dataPointsWithClassC[c])/float(len(df.index))

        for d in self.attributes:
            for c in self.classes:
                self.classToAttributeMeans[c][d] = dataPointsWithClassC[c][d].mean()
                self.classToAttributeVariances[c][d] = dataPointsWithClassC[c][d].var()

        # self.summarizeInputData()
        self.trained = True

    def pdfOfInputDataPointForClassAndAttribute(self, c, d, dValue, pdf):
        return pdf(self.classToAttributeMeans[c][d], self.classToAttributeVariances[c][d], dValue)

    def logLikelihoodOfInputDataPointGivenClass(self, c, df):
        sum = 0
        for d in self.attributes:
            prior = self.classLikelihood[c]
            conditional =  self.pdfOfInputDataPointForClassAndAttribute(c, d, df[d], self.normalPDF)
            joint = prior * conditional
            if joint > 0:
                sum += math.log(joint)
        return sum

    def normalizeSinglePoint(self, inputFrame):
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

    def classify(self, inputFrame):
        assert self.trained

        unknown = self.normalizeSinglePoint(inputFrame)

        bestLogLikelihood = 0
        bestClassification = None
        for c in self.classes:
            log_likelihood = self.logLikelihoodOfInputDataPointGivenClass(c, unknown)
            if bestLogLikelihood < log_likelihood or bestClassification is None:
                bestLogLikelihood = log_likelihood
                bestClassification = c
        return bestClassification
