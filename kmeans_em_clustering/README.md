***KMEANS, KERNEL KMEAN, EM FOR GAUSSIAN MIXTURE MODEL***

The tests in this repo run kmean clustering, kernel kmeans clustering and EM under GMM assumptions on two different data sets. The concentric circles dataset must be taken to a new feature space to appropriately clustered, and the EM algorithm is not run on this data set.

Run `python run.py` to generate a set of images that demonstrate the progress of the algorithm under different parameters.

***INTERESTING NOTES***

It is interesting to run k > 3 kernel kmeans clustering algorithms on the concentric circles data set. and to watch the colors change as the radius increases!

In the `images/loglikelihood.png`, we see the loglikelihood of 5 seperate (random) runs of the EM algorithm on the blob data set. Note that the values are monotonically increasing (yay!). Also, it seems like I may be adding by some irrelevant/incorrect constant factor though.
