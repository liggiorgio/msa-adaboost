import numpy as np
import pandas as pd

from adaboost import AdaBoost

# Custom implementation of the AdaBoost multiclass classifier wrapper.

class AdaBoostM1:

    # Initialise AdaBoost multiclass classifier
    def __init__(self, boosting_rounds = 50):
        self.classes = 0
        self.classifiers = None
        self.T = boosting_rounds


    # Train the multiclass classifier
    def fit(self, X, y):
        self.classes = np.unique(y)

        self.classifiers = {}
        y_class = {}

        # Define one-vs-all DataFrames, encoding for {-1, 1} labels
        for c in self.classes:
            y_class[c] = (y == c).astype(int) * 2 - 1

        # Define ensemble classifiers for each class and train it
        for c in self.classes:
            self.classifiers[c] = AdaBoost(boosting_rounds = self.T)
            self.classifiers[c].fit(X, y_class[c])
            #print('=', end = '')
    

    # Predict multiclass labels from a feature set
    def predict(self, X, depth = -1):
        y_pred = pd.DataFrame(index = X.index, columns = self.classes)
        
        for c in self.classes:
            y_pred[c] = self.classifiers[c].predict(X, depth = depth)
            if (depth == -1):
                print(c, "done")
        
        return y_pred.idxmax(axis = 'columns')


    # Get decision features
    def get_features(self, X):
        labels = []
        for c in self.classes:
            labels.append( self.classifiers[c].get_features(X) )
        return labels