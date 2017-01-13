import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataFile = "housing.data"
columnHeaders = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
featureHeaders = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
everyNthDatapointIsTest = 7

def splitEveryNthToTest(df, n):
    test = df[::n]
    train = df[~df.isin(test).all(1)]
    return (train, test)

def loadCSV(filename, headers):
    return pd.pandas.read_csv(filename, names=headers, delim_whitespace=True)

def getNormalizedDF(df, headers):
    normalized = df.copy()
    normalized[headers] = (normalized[headers] - normalized[headers].mean()) / normalized[headers].std()
    return normalized

def getWeightsAnalytically(designMatrix, yn):
    return np.dot(np.dot(np.linalg.pinv(np.dot(designMatrix.T, designMatrix)), designMatrix.T), yn)

def getWeightsNumerically(designMatrix, yn, stepSize, maxError, batch=True):
    if batch:
        return batchGradientDescent(designMatrix, yn, stepSize, maxError)
    else:
        return stochasticGradientDescent(designMatrix, yn, stepSize, maxError)

def stochasticGradientDescent(designMatrix, yn, stepSize, maxError):
    import random
    w = np.ones(14)
    while True:
        r = random.randint(0, len(yn)-1)
        w_new = w - stepSize * (np.dot(np.dot(designMatrix[r].T, w) -  yn[r], designMatrix[r]))
        if np.absolute(w - w_new).max() < maxError:
            break
        w = w_new
    return w

def batchGradientDescent(designMatrix, yn, stepSize, maxError):
    w = np.ones(14)
    while True:
        w_new = w - stepSize * (np.dot(np.dot(designMatrix.T, designMatrix), w) - np.dot(designMatrix.T, yn))
        if np.absolute(w - w_new).max() < maxError:
            break
        w = w_new
    return w

def runLinearRegression(train, test):
    means = train.mean()[featureHeaders]
    stdDevs = train.std()[featureHeaders]
    normalizedTraining = getNormalizedDF(train, featureHeaders)

    ones = np.array([np.ones(len(normalizedTraining))])
    designMatrix = np.concatenate((ones.T, normalizedTraining[featureHeaders].values), axis=1)
    yn = normalizedTraining["MEDV"].values

    wLMS = getWeightsAnalytically(designMatrix, yn)
    # wLMS = getWeightsNumerically(designMatrix, yn, stepSize = .001, maxError = .0000001)

    print "MSE on training data: {}".format(mseForWeightsAndData(train, featureHeaders, means, stdDevs, wLMS))
    print "MSE on test data: {}\n".format(mseForWeightsAndData(test, featureHeaders, means, stdDevs, wLMS))

def mseForWeightsAndData(data, headers, means, stdDevs, weights):
    mseSum = 0
    for index, row in data.iterrows():
        value = row["MEDV"]
        normalizedRow = (row[headers] - means) / stdDevs
        x = np.append([1], normalizedRow.values)
        mseSum += (value - np.dot(x, weights)) ** 2
    import math
    return mseSum/float(len(data))

def ridgeRegressionWeightsAndMse(train, test, A):
    means = train.mean()[featureHeaders]
    stdDevs = train.std()[featureHeaders]
    normalizedTraining = getNormalizedDF(train, featureHeaders)

    ones = np.array([np.ones(len(normalizedTraining))])
    designMatrix = np.concatenate((ones.T, normalizedTraining[featureHeaders].values), axis=1)
    yn = normalizedTraining["MEDV"].values

    wMap = np.dot(np.linalg.pinv(np.dot(designMatrix.T, designMatrix) + np.dot(A, np.identity(len((designMatrix.T))))), np.dot(designMatrix.T, yn))
    return wMap, mseForWeightsAndData(test, featureHeaders, means, stdDevs, wMap)

def getKFoldsFromTraining(train, k):
    shuffled = train.sample(frac=1).reset_index(drop=True)
    foldSize = len(shuffled)/k
    groups = []
    for i in range(0, k):
        if i != (k - 1):
            groups.append(shuffled.iloc[i*foldSize:(i+1)*foldSize])
        else:
            groups.append(shuffled.iloc[i*foldSize:])
    return groups

def runRidgeRegression(train, test):
    groups = getKFoldsFromTraining(train, 10)

    # For plotting and tracking best lambda
    AList = []
    trainMSE = []
    testMSE = []
    bestCVA = None
    bestCVMse = None
    bestTestA = None
    bestTestMse = None

    minA = .0001
    maxA = 10
    trials = 200
    A = minA
    while A <= maxA:
    # for A in [.0001, .001, .01, .1, 1, 10]:
        AList.append(A)
        mseForThisA = 0
        for g in groups:
            thisTrain = train[~train.isin(g).all(1)]
            (wMap, cvMse) = ridgeRegressionWeightsAndMse(thisTrain, g, A)
            mseForThisA += cvMse
        avgCVMse = mseForThisA/float(len(groups))

        if False:
            print "\nLAMDBA = {}".format(A)
            print "---------------".format(A)
            print "Average MSE with 10-fold CV:\t{}".format(avgCVMse)

        trainMSE.append(avgCVMse)
        if bestCVMse is None or avgCVMse < bestCVMse:
            bestCVMse = avgCVMse
            bestCVA = A

        if False:
            (wMap, testMse) = ridgeRegressionWeightsAndMse(train, test, A)
            print "MSE on test set:\t\t{}\n".format(testMse)
            testMSE.append(testMse)
            if bestTestMse is None or testMse < bestTestMse:
                bestTestA = A
                bestTestMse = testMse

        A += (maxA - minA) / trials

    print "Lambda with lowest MSE on CV Data:\tlambda = {},\t\tMSE = {}".format(bestCVA, bestCVMse)
    (wMap, mseOnTest) = ridgeRegressionWeightsAndMse(train, test, bestCVA)
    print "MSE of best lambda on test set:\t\t{}\n".format(mseOnTest)
    # print "Lambda with lowest MSE on test Data:\tlambda = {},\tMSE = {}".format(bestTestA, bestTestMse)

    fig = plt.figure()
    plt.plot(AList, trainMSE, 'ro')
    fig.suptitle("Lambda vs MSE")
    plt.savefig("lambda_vs_mse.png")
    plt.close()

    print "\nGraph of lambda vs MSE printed at lambda_vs_mse.png.\n"

def runFeatureSelection(train, test):
    # PART A: TOP CORRELATED FEATURES
    print "\nPart 3.3 a - Select features with best correlation to target"
    print "--------------------------------------------------------------\n"
    reducedHeaders = ["LSTAT", "PTRATIO", "RM", "TAX"]
    print "Features with highest pearson coefficient to target:\n{}".format(reducedHeaders)
    means = train.mean()[reducedHeaders]
    stdDevs = train.std()[reducedHeaders]
    normalizedTraining = getNormalizedDF(train, reducedHeaders)

    ones = np.array([np.ones(len(normalizedTraining))])
    designMatrix = np.concatenate((ones.T, normalizedTraining[reducedHeaders].values), axis=1)
    yn = normalizedTraining["MEDV"].values

    wLMS = getWeightsAnalytically(designMatrix, yn)
    mse = mseForWeightsAndData(test, reducedHeaders, means, stdDevs, wLMS)
    print "MSE on test set {}\n".format(mse)

    # PART B: HIGHEST CORRELATION WITH RESIDUAL
    print "\nPart 3.3 b - Select features by correlation with the residual"
    print "------------------------------------------------------------\n"
    residual = yn
    unselectedFeatures = list(featureHeaders)
    selectedHeaders = []
    for _ in range(4):

        # calculate feature with highest correlation to residual
        bestFeature = None
        bestCoef = None
        for feature in unselectedFeatures:
            df = pd.DataFrame({feature: train[feature], "residual": residual})
            pearson = df.cov()[feature]["residual"]/(df[feature].std() * df["residual"].std())
            if bestCoef is None or pearson < bestCoef:
                bestFeature = feature
                bestCoef = pearson

        unselectedFeatures.remove(bestFeature)
        correlatedFeature = bestFeature
        selectedHeaders.append(correlatedFeature) 
        print "Selecting feature {} with pearson correlation {} to the residual.".format(correlatedFeature, bestCoef)

        # Get weights using selected headers
        means = train.mean()[selectedHeaders]
        stdDevs = train.std()[selectedHeaders]
        normalizedTraining = getNormalizedDF(train, selectedHeaders)

        ones = np.array([np.ones(len(normalizedTraining))])
        designMatrix = np.concatenate((ones.T, normalizedTraining[selectedHeaders].values), axis=1)
        yn = normalizedTraining["MEDV"].values
        weights = getWeightsAnalytically(designMatrix, yn)

        # get residual from weights
        residual = (yn - np.dot(designMatrix, weights))

    mse = mseForWeightsAndData(test, selectedHeaders, means, stdDevs, weights)
    print "MSE on test set for selected headers {}: {}\n".format(selectedHeaders, mse)

    # PART C: BRUTE FORCE
    print "Part 3.3 - Brute force search for best 4 features"
    print "-------------------------------------------------\n"
    bestMSE = None
    bestParameters = None
    for i in range(0, len(featureHeaders)-1):
        for j in range(i+1, len(featureHeaders)-1):
            for k in range(j+1, len(featureHeaders)-1):
                for l in range(k+1, len(featureHeaders)-1):
                    reducedHeaders = [featureHeaders[i], featureHeaders[j], featureHeaders[k], featureHeaders[l]] 

                    means = train.mean()[reducedHeaders]
                    stdDevs = train.std()[reducedHeaders]
                    normalizedTraining = getNormalizedDF(train, reducedHeaders)

                    ones = np.array([np.ones(len(normalizedTraining))])
                    designMatrix = np.concatenate((ones.T, normalizedTraining[reducedHeaders].values), axis=1)
                    yn = normalizedTraining["MEDV"].values

                    wLMS = getWeightsAnalytically(designMatrix, yn)
                    mse = mseForWeightsAndData(test, reducedHeaders, means, stdDevs, wLMS)
                    if bestMSE is None or mse < bestMSE:
                        bestMSE = mse
                        bestParameters = reducedHeaders

    print "Best set of 4 parameters found with brute force:\n{}\nMSE on test set: {}\n".format(bestParameters, bestMSE)

def runPolynomialFeatureExpansion(train, test):
    expandedTrain = train.copy()
    expandedTest = test.copy()
    expandedHeaders = list(featureHeaders)
    columnName = 0
    for i in range(len(featureHeaders)):
        for j in range(i, len(featureHeaders)):
            expandedTrain[str(columnName)] = expandedTrain[featureHeaders[i]] * expandedTrain[featureHeaders[j]]
            expandedTest[str(columnName)] = expandedTest[featureHeaders[i]] * expandedTest[featureHeaders[j]]
            expandedHeaders.append(str(columnName))
            columnName += 1

    means = expandedTrain.mean()[expandedHeaders]
    stdDevs = expandedTrain.std()[expandedHeaders]
    normalizedTraining = getNormalizedDF(expandedTrain, expandedHeaders)

    ones = np.array([np.ones(len(normalizedTraining))])
    designMatrix = np.concatenate((ones.T, normalizedTraining[expandedHeaders].values), axis=1)
    yn = normalizedTraining["MEDV"].values

    wLMS = getWeightsAnalytically(designMatrix, yn)
    mse = mseForWeightsAndData(expandedTrain, expandedHeaders, means, stdDevs, wLMS)
    print "\nPolynomail Feature Expansion MSE on Training: {}".format(mse)
    mse = mseForWeightsAndData(expandedTest, expandedHeaders, means, stdDevs, wLMS)
    print "Polynomail Feature Expansion MSE on Testing: {}\n".format(mse)
    runRidgeRegression(expandedTrain, expandedTest)

def main():
    # Load Data
    df = loadCSV(dataFile, columnHeaders)
    train, test = splitEveryNthToTest(df, everyNthDatapointIsTest)

    print "\n#####################"
    print "# Linear Regression #"
    print "#####################\n"
    runLinearRegression(train, test)

    print "\n#####################"
    print "# Ridge Regression #"
    print "#####################\n"
    runRidgeRegression(train, test)

    print "\n#####################"
    print "# Feature Selection #"
    print "#####################"
    runFeatureSelection(train, test)

    print "\n################################"
    print "# Polynomial Feature Expansion #"
    print "################################"
    runPolynomialFeatureExpansion(train, test)

if __name__ == "__main__":
    main()
