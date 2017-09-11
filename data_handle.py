import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

def read_data(url):
    data = pd.read_csv(url)
    dummies = pd.get_dummies(data['group'], prefix='group', drop_first=False)
    label = data['label']
    weight = data['weight']
    X = data.drop(['group','era','id','weight','label'],axis=1)
    return X, dummies, weight, label

def scale_feature(X,dummies,quantile_percent=0.9):
    scaled_features = {}
    for each in X.columns:
        mean, std = X[each].mean(), X[each].std()
        scaled_features[each] = [mean,std]
        X.loc[:, each] = (X[each] - mean)/std
        X.loc[X[each]>X[each].quantile(quantile_percent)] = X[each].quantile(quantile_percent)
    X = pd.concat([X, dummies], axis=1)
    return X, scaled_features

def data_split(X,Y,test_size=0.2):
    values = X.values
    labels = Y.values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    for train_index, test_index in sss.split(values,labels):
        X_train, Y_train = values[train_index], labels[train_index]
        X_test, Y_test = values[test_index], labels[test_index]
    return X_train, Y_train, X_test, Y_test