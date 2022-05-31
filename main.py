from collections import Counter
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import zero_one_loss

from adaboostm1 import AdaBoostM1


if __name__ == '__main__':
    # Define constants
    DATAPATH = './covtype.csv'
    N_SPLITS = 10
    N_ESTIMATORS = 50
    RANDOM_STATE = 42


    # Import dataset
    data = pd.read_csv(DATAPATH)
    data = data.head(15120)


    # Split DataFrame
    X = data.drop(['Cover_Type'], axis = 1)     #features
    y = data['Cover_Type']                      #labels


    # Data export
    output = pd.DataFrame(index = range(1, N_ESTIMATORS + 1), columns = ['loss_tr', 'accuracy_tr', 'loss_te', 'accuracy_te'])
    features = []

    # Perform CV over T boosting rounds for N-fold sets
    kfold = KFold(n_splits = N_SPLITS, shuffle = True, random_state = RANDOM_STATE)

    for train_index, test_index in kfold.split(X, y):
        
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        classifier = AdaBoostM1(boosting_rounds = N_ESTIMATORS)
        classifier.fit(X_train, y_train)

        temp = pd.DataFrame(index = range(1, N_ESTIMATORS + 1), columns = ['loss_tr', 'accuracy_tr', 'loss_te', 'accuracy_te'])

        for round in range(1, N_ESTIMATORS + 1):
            y_pred_train = classifier.predict(X_train, depth = round)
            y_pred = classifier.predict(X_test, depth = round)

            loss_train = zero_one_loss(y_train, y_pred_train)
            loss_test = zero_one_loss(y_test, y_pred)
            print(round, '-', loss_train, 1 - loss_train, loss_test, 1 - loss_test)
            temp.loc[round] = [loss_train, 1 - loss_train, loss_test, 1 - loss_test]
        
        output = output.add(temp, fill_value = 0)
        print()

        features.append( classifier.get_features(X_train) )

    output['loss_tr'] = output['loss_tr'] / N_SPLITS
    output['accuracy_tr'] = output['accuracy_tr'] / N_SPLITS
    output['loss_te'] = output['loss_te'] / N_SPLITS
    output['accuracy_te'] = output['accuracy_te'] / N_SPLITS
    features = np.reshape(features, -1)

    cwd = os.getcwd()
    path = cwd + '/res/mean_loss-' + str(N_SPLITS) + 'f_' + str(N_ESTIMATORS) + 'e_' + str(RANDOM_STATE) + 's.csv'
    output.to_csv(path)

    path2 = cwd + '/res/feature_counter-' + str(N_SPLITS) + 'f_' + str(N_ESTIMATORS) + 'e_' + str(RANDOM_STATE) + 's.csv'
    pd.Series(dict(Counter(features))).to_csv(path2)
