import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

imageDirectory = "images"

degreeToLabel = {
    -1: "g1",
    0: "g2",
    1: "g3",
    2: "g4",
    3: "g5",
    4: "g6",
}

class LinearRegression:
    degree = -1
    weights = [1]

    def __init__(self, degree):
        self.degree = degree

    def train(self, data, A=0):
        if self.degree == -1:
            return
        X = np.array([[sample ** i for i in range(0, self.degree + 1)] for sample in data["x"]])
        designMatrix = np.dot(X.T, X)
        regularizingComponent = A * np.identity(len(X.T))
        inverted = np.linalg.pinv(designMatrix + regularizingComponent)
        self.weights = np.dot(np.dot(inverted, X.T), data["y"])

    def G(self, x):
        if self.degree == -1:
            return 1
        return sum([self.weights[i] * x ** i for i in range(self.degree + 1)])

def generateSample(pointsPerSample, xLow, xHigh):
    d = pd.DataFrame({"x": np.random.uniform(-1, 1, pointsPerSample)})
    d["y"] = (2 * d["x"] ** 2) + np.random.normal(0, math.sqrt(.1), pointsPerSample)
    return d

def createImageDirectory(imageDirectory):
    if not os.path.exists(imageDirectory):
        os.makedirs(imageDirectory)

def calculateMSE(data, model):
    mse = 0
    for example in data.values:
        mse += (model.G(example[0]) - example[1]) ** 2
    return mse / float(len(data.values))

def averageMSEOverSamples(samples, model):
    avg = 0
    for s in samples:
        avg += calculateMSE(s, model)
    return avg / float(len(samples))

def avgG(functions, x):
    return np.mean([f.G(x) for f in functions])

def calculateBiasVariance(xLow, xHigh, xIterations, functions):
    bias = 0.0
    variance = 0.0
    probX = 1/float(xHigh - xLow)
    dx = (xHigh - xLow)/float(xIterations)
    for i in range(xIterations):
        xVal = xLow + i * dx
        expectedY = 2 * (xVal ** 2)
        bias += (avgG(functions, xVal) - expectedY) ** 2
        variance += np.var([g.G(xVal) for g in functions])

    bias *= dx * probX
    variance *= dx * probX

    return bias, variance

def mseHistogram(mses, modelName, pointsPerSample, imageDirectory):
    fig = plt.figure()
    fig.suptitle("")
    fig.suptitle('MSE Histogram {}: {} Samples'.format(modelName, pointsPerSample), fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel('MSE')
    plt.hist(mses)
    fig.savefig(imageDirectory + "/{}_histogram_{}".format(pointsPerSample, modelName))
    plt.close()

def biasVarianceExperiment():
    xLow = -1
    xHigh = 1
    xIterations = 100
    numberOfSamples = 100
    # pointsPerSample = 10
    samples = []
    degreeToFunctions = {}
    degreeToMSEs = {}
    maxDegrees = 4
    createImageDirectory(imageDirectory)


    for pointsPerSample in [10, 100]:
        samples = []
        for x in range(numberOfSamples):
            samples.append(generateSample(pointsPerSample, xLow, xHigh))

        biases = []
        variances = []
        mses = []
        for d in range(-1, maxDegrees + 1):
            degreeToFunctions[d] = []
            degreeToMSEs[d] = []

            for s in samples:
                g = LinearRegression(d)
                g.train(s)
                degreeToMSEs[d].append(calculateMSE(s, g))
                degreeToFunctions[d].append(g)

            mseHistogram(degreeToMSEs[d], degreeToLabel[d], pointsPerSample, imageDirectory)
            bias, variance = calculateBiasVariance(xLow, xHigh, xIterations, degreeToFunctions[d])
            avgMSE = sum(degreeToMSEs[d])/float(len(degreeToMSEs[d]))
            biases.append(bias)
            variances.append(variance)
            mses.append(avgMSE)

            print "{} Samples - {}".format(pointsPerSample, degreeToLabel[d])
            print "-----------"
            print "BIAS: {}".format(bias)
            print "VARIANCE: {}".format(variance)
            print "MSE: {}".format(avgMSE)
            print

        # PLOT BIAS AND VARIANCE VS MODEL COMPLEXITY
        # Model Complexity VS BIAS
        fig = plt.figure()
        fig.suptitle("")
        fig.suptitle('Model Complexity vs Bias: {} Samples'.format(pointsPerSample), fontsize=14, fontweight='bold')
        ax = fig.add_subplot(111)
        ax.set_xlabel('Model Complexity')
        ax.set_ylabel('Bias')
        plt.plot(range(-1, maxDegrees + 1), biases)
        fig.savefig(imageDirectory + "/{}perSample_bias".format(pointsPerSample))
        plt.close()

        # Model Complexity VS VARIANCE
        fig = plt.figure()
        fig.suptitle("")
        fig.suptitle('Model Complexity vs Variance: {} Samples'.format(pointsPerSample), fontsize=14, fontweight='bold')
        ax = fig.add_subplot(111)
        ax.set_xlabel('Model Complexity')
        ax.set_ylabel('Variance')
        plt.plot(range(-1, maxDegrees + 1), variances)
        fig.savefig(imageDirectory + "/{}perSample_variance".format(pointsPerSample))
        plt.close()

        # Model Complexity VS MSE
        fig = plt.figure()
        fig.suptitle("")
        fig.suptitle('Model Complexity vs MSE: {} Samples'.format(pointsPerSample), fontsize=14, fontweight='bold')
        ax = fig.add_subplot(111)
        ax.set_xlabel('Model Complexity')
        ax.set_ylabel('MSE')
        plt.plot(range(-1, maxDegrees + 1), mses)
        fig.savefig(imageDirectory + "/{}perSample_mse".format(pointsPerSample))
        plt.close()

    numberOfSamples = 100
    samples = []
    for x in range(numberOfSamples):
        samples.append(generateSample(pointsPerSample, xLow, xHigh))

    # PLOT BIAS AND VARIANCE VS LAMBDA
    biases = []
    variances = []
    mses = []
    As = [.001, .003, .01, .03, .1, .3, 1]
    for A in [.001, .003, .01, .03, .1, .3, 1]:
        functions = []
        for s in samples:
            g2 = LinearRegression(2)
            g2.train(s, A=A)
            functions.append(g2)
        bias, variance = calculateBiasVariance(xLow, xHigh, xIterations, functions)
        biases.append(bias)
        variances.append(variance)
        mses.append(averageMSEOverSamples(samples, g2))

    # REGULARIZATION VS BIAS
    fig = plt.figure()
    fig.suptitle("")
    fig.suptitle('Regularization vs Bias', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel('Regularization')
    ax.set_ylabel('Bias')
    plt.plot(As, biases)
    fig.savefig(imageDirectory + "/lambda_vs_bias")
    plt.close()

    # REGULARIZATION VS VARIANCE
    fig = plt.figure()
    fig.suptitle("")
    fig.suptitle('Regularization vs Variance', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel('Regularization')
    ax.set_ylabel('Variance')
    plt.plot(As, variances)
    fig.savefig(imageDirectory + "/lambda_vs_variance")
    plt.close()

    # REGULARIZATION VS MSE
    fig = plt.figure()
    fig.suptitle("")
    fig.suptitle('Regularization vs MSE', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel('Regularization')
    ax.set_ylabel('MSE')
    plt.plot(As, mses)
    fig.savefig(imageDirectory + "/lambda_vs_mse")
    plt.close()

if __name__ == "__main__":
    biasVarianceExperiment()
