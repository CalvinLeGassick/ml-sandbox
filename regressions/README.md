***Regression Tests***

This repository contains some simple experiements of running hand built regression algorithms on the UCI Housing Data Set (archive.ics.uci.edu/ml/datasets/Housing).

The code here explores:
- Linear & Ridge regression with the analytic solution as well as batch and stochastic gradient descent.
- Linear regression over all 4-tuples of the feature set to find which subset of features can be used to produce the best results.
- Linear regression on polynomial expansions of the feature space.

Hyper parameters for ridge regression are trained using 10-fold cross validation. Produced graph (`lambda_vs_mse`) tends to indicate that we may not need a regularization term for this data set. Should compare with results from scikit-learn.

Running `python run` will train and evaluate all classifiers.

***Notes from Experiments***

Polynomial expansion performs incredibly here compared to the other methods.

***TODO***

- Modularization into regression.py and other files.
- Compare results with scikit-learn to verify correctness.
- Compare implementation with scikit-learn and other implementations to understand better/alternate coding style.
- Visualizations of performance of methods.