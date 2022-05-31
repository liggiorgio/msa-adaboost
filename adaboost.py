import os
from collections import Counter
import numpy as np
from sklearn.tree import DecisionTreeClassifier


# Custom implementation of the AdaBoost binary ensemble.


class AdaBoost:
    
    # Initialise AdaBoost classifier
    def __init__(self, boosting_rounds = 50):
        self.T = boosting_rounds            #no. of estimators - Int
        self.learners = [None] * self.T     #weak classifiers & alphas - Tuple


    # Train the classifier
    def fit(self, X, y):
        w_i = np.ones(len(y)) / len(y)  #init sample weights at 1/n

        for t in range(self.T):
            G_t = DecisionTreeClassifier(max_depth = 1)

            G_t.fit(X, y, sample_weight = w_i)
            y_pred = G_t.predict(X)

            err_t = compute_error(y, y_pred, w_i)
            alpha_t = compute_alpha(err_t)
            w_i = update_weights(y, y_pred, w_i, alpha_t)

            self.learners[t] = (G_t, alpha_t)
            #print(alpha_t)
            #print(t, X.columns[G_t.tree_.feature[0]] )


    # Predict labels from a feature set
    def predict(self, X, depth = -1):
        if depth < 1:
            depth = self.T

        pred = np.zeros(len(X))

        for t in range(depth):
            G_t, alpha_t = self.learners[t]
            pred += G_t.predict(X) * alpha_t
        
        return pred

    
    # Get decision features
    def get_features(self, X):
        labels = []
        for t in range(self.T):
            G_t, _ = self.learners[t]
            labels.append( X.columns[G_t.tree_.feature[0]] )
        return labels


# Indicator function
def I(input):
    return input.astype(int)


# Compute the prediction error
def compute_error(y, y_pred, weights):
    return sum( weights * I(y != y_pred) ) / sum(weights)


# Compute the weak learner amount of say
def compute_alpha(error):
    return 0.5 * np.log( (1 - error) / error )


# Compute new sample weights
def update_weights(y, y_pred, weights, alpha):
    new_weights = weights * np.exp(-alpha * y * y_pred)
    return new_weights / sum(new_weights)
