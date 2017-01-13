import random
import math
from collections import OrderedDict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plotColors = ['ro', 'bo', 'go', 'wo', 'yo']
markerColors = ['red', 'blue', 'green', 'black', 'yellow']

plots = 0

def findMeanForPoint(example, means):
    bestMeanVal = None
    bestMean = None
    for index, m in means.iterrows():
        distance = ((example[means.columns] - m) ** 2).sum()

        if bestMean is None or distance < bestMeanVal:
            bestMean = index
            bestMeanVal = distance

    return bestMean

def graphClusterAssignments(assignments, means, k, imageDirectory, title, graphCenters):
    global plots
    fig = plt.figure()
    fig.suptitle('{} {}-Means Clustering'.format(title, k), fontsize=14, fontweight='bold')
    for l in range(len(means)):
        plt.plot([a[0] for a in assignments[l]], [a[1] for a in assignments[l]], plotColors[l])

    for l in range(len(means)):
        if graphCenters:
            plt.plot(means["x"][l], means["y"][l], 'x', color=markerColors[l], mew=3, ms=10)

    fig.savefig(imageDirectory + "/{}_{}-means{}.png".format(title, k, plots))
    plt.close()
    plots += 1

def getAssignments(df, means):
    assignments = {}
    for index, m in means.iterrows():
        assignments[index] = []

    for index, row in df.iterrows():
        mean = findMeanForPoint(row, means)
        assignments[mean].append(row)

    return assignments

def runKMeans(inputFile, imageDirectory, k=2):
    df = pd.read_csv(inputFile, names=["x", "y"])

    init_means = [(random.choice(df["x"]), random.choice(df["y"])) for _ in range(k)]
    means = pd.DataFrame(init_means, columns=["x", "y"])

    new_means = None
    assignments = getAssignments(df, means)
    while new_means is None or not means.equals(new_means):
        means = means if new_means is None else new_means
        try:
            new_means = pd.DataFrame([np.mean(assignments[m], (0)) if len(assignments[m]) else means.loc(0)[m].values for m in range(len(means))], columns=["x", "y"])
        except Exception as e:
            import pdb; pdb.set_trace()
        assignments = getAssignments(df, new_means)
        graphClusterAssignments(assignments, new_means, k, imageDirectory, inputFile.split("_")[1].split(".")[0], True)

    return new_means

def runKernelMeans(inputFile, imageDirectory, k=2):
    df = pd.read_csv(inputFile, names=["x", "y"])
    df["z"] = df["x"] ** 2 + df["y"] ** 2

    init_means = [(random.choice(df["z"])) for _ in range(k)]
    means = pd.DataFrame(init_means, columns=["z"])

    new_means = None
    assignments = getAssignments(df, means)
    while new_means is None or not means.equals(new_means):
        means = means if new_means is None else new_means
        new_means = pd.DataFrame([np.mean([l["z"] for l in assignments[m]], (0)) for m in range(len(means))], columns=["z"])
        assignments = getAssignments(df, new_means)
        graphClusterAssignments(assignments, new_means, k, imageDirectory, "kernel_" + inputFile.split("_")[1].split(".")[0], False)

def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    sigma = np.matrix(sigma)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0 / ( math.pow((2*math.pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = np.matrix(x - mu)
        inv = sigma.I
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")

def E_STEP(df, mus, sigmas, priors):
    gammas = OrderedDict({})
    for i in range(len(mus)):
        gammas[i] = OrderedDict({})

    for index, row in df.iterrows():
        normalizer = sum( [ norm_pdf_multivariate(row.values, mus[k], sigmas[k]) * priors[k] for k in range(len(mus)) ] )
        for k in range(len(mus)):
            gammas[k][tuple(row.values)] = norm_pdf_multivariate(row.values, mus[k], sigmas[k]) * priors[k] / normalizer

    return gammas

def updatePriors(gammas):
    priors = OrderedDict({})
    totalExamples = sum( [ sum(gammas[k].values()) for k in gammas ] )
    for k, v in gammas.iteritems():
       priors[k] = sum(v.values()) / totalExamples

    return priors

def M_STEP(df, gammas):
    priors = updatePriors(gammas)
    dimensions = len(df.loc(0)[0].values)

    mus = OrderedDict({})
    sigmas = OrderedDict({})

    for k in range(len(gammas)):
        denominator = 1 / float(sum([gammas[k][tuple(row.values)] for _, row in df.iterrows()]))
        mus[k] = np.multiply( denominator, sum([gammas[k][tuple(row.values)] * row.values for _, row in df.iterrows()]) )
        sigmas[k] = np.multiply( denominator,
            sum([
                gammas[k][tuple(row.values)]
                * np.matrix((mus[k] - row.values)).T
                * np.matrix((mus[k] - row.values))
                for _, row in df.iterrows()
            ])
        )

    return mus, sigmas, priors

def calculate_loglikelihood(df, mus, sigmas, gammas):
    total = 0.0
    for index, row in df.iterrows():
        bestCluster, bestClusterProbability = bestAssignmentForExample(row, gammas)
        total += math.log(sum([ norm_pdf_multivariate(row.values, mus[k], sigmas[k]) * gammas[k][tuple(row.values)] for k in gammas]), 2)

    return total

def plotLoglikelihood(likelihoods, imageDirectory):
    fig = plt.figure()
    fig.suptitle("Log Likelihood after each EM Iteration", fontsize=14, fontweight='bold')
    for l in likelihoods:
        plt.plot(range(len(l)), l)
    fig.savefig(imageDirectory + "/loglikelihoods.png")
    plt.close()

def runGMM(inputFile, imageDirectory, k=2):
    df = pd.read_csv(inputFile, names=["x", "y"])

    totalLikelihoods = []

    bestL = None
    bestLs = None
    bestMus = None
    bestCovs = None
    bestGammas = None

    for _ in range(5):
        print "\nGAUSSIAN MIXTURE MODEL CLUSTERING"
        # print "selecting initial means with k-means clustering..."
        mu_array = [(random.choice(df["x"]), random.choice(df["y"])) for _ in range(k)]
        # mu_array = runKMeans(inputFile, imageDirectory, k).values
        priors = OrderedDict({})
        mus = OrderedDict({})
        sigmas = OrderedDict({})
        for k_ in range(k):
            priors[k_] = 1 / float(k)
            mus[k_] = mu_array[k_]
            sigmas[k_] = np.matrix(df.cov().values)

        print "Initializing EM...\n"
        likelihoods = []
        for i in range(5):
            gammas = E_STEP(df, mus, sigmas, priors)
            mus, sigmas, priors = M_STEP(df, gammas)
            loglikelihood = calculate_loglikelihood(df, mus, sigmas, gammas)
            likelihoods.append(loglikelihood)
            print "LOG LIKELIHOOD {} ITERATION: {}".format(i, loglikelihood)

        totalLikelihoods.append(likelihoods)

        if bestL is None or likelihoods[-1] > bestL:
            bestL = likelihoods[-1]
            bestLs = likelihoods
            bestMus = mus
            bestCovs = sigmas
            bestGammas = gammas

    print "\nLOGLIKELIHOOD: {}".format(bestL)
    print "\nMEANS"
    print "\n".join(["{}".format(m) for m in bestMus.values()])
    print "\nCOVARS"
    print "\n".join(["{}".format(s) for s in bestCovs.values()])
    print
    means = pd.DataFrame(bestMus.values(), columns=["x", "y"])
    assignments = getGMMAssignments(df, bestGammas)
    graphClusterAssignments(assignments, means, k, imageDirectory, "test{}".format(i), True)

    plotLoglikelihood(totalLikelihoods, imageDirectory)

def getGMMAssignments(df, gammas):
    assignments = OrderedDict({})
    for k in gammas.keys():
        assignments[k] = []

    for index, row in df.iterrows():
        bestCluster, _ = bestAssignmentForExample(row, gammas)
        assignments[bestCluster].append(row)

    return assignments

def bestAssignmentForExample(row, gammas):
    bestCluster = None
    bestClusterProbability = None
    for k in gammas:
        clusterProbability = gammas[k][tuple(row.values)]
        if bestCluster is None or clusterProbability > bestClusterProbability:
            bestCluster = k
            bestClusterProbability = clusterProbability

    return bestCluster, bestClusterProbability
