***KNN + Naive Bayes Tests***

This repository contains some simple experiements of running hand buily knn and naive bayes classifiers on the USC Glass Identification dataset (archive.ics.uci.edu/ml/datasets/Glass+Identification).

Running `python run` will train and evaluate both classifiers.

***Notes from Experiments***

Naive Bayes classifier performs terribly. Need to compare with other implementations / debug for a bit to see, but this could also be because the naive bayes assumption just does not apply here.

KNN seems too slow. Need to check around for a bit myself to speed up, and then check other implementations if I don't spot the slowdown.

***IMPORTANT***

Hyper parameters are evaluated against the test data for the KNN code. **DO NOT DO THIS IN PRACTICE**. The proper pipeline is to further split up the training data into `train` and `dev` (also called `validation`) data sets, and to evaluate the hyper parameters on the dev data set. Then test should be evaluated only on final implementations to get an understanding of generalization ability.

***TODO***

- Compare results with scikit-learn to verify correctness.
- Compare implementation with scikit-learn and other implementations to understand better/alternate coding style.
- Implement dev pipeline (?)