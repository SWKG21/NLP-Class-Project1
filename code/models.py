import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import  MLPClassifier
from xgboost import XGBClassifier
import tensorflow as tf



def scaling(X_train, X_test, Standard=False):
    """
        Given the training set and the test set, 
        use a StandardScaler or MinMaxScaler to do scaling, 
        return the two scaled datasets
    """
    if Standard:
        scaler = StandardScaler()
        
    else:
        scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test
    

def CV(classifier, X, y, nfold=5):
    """
        Given the classifier and the datasets, 
        do cross validation and calculate f1 scores.
    """
    X, y = shuffle(X, y, random_state=0)
    return cross_val_score(classifier, X, y, cv=nfold, scoring='f1').mean()


def trainAndPredict(classifier, X_train, y_train, X_test, save_name, path):
    """
        Given the classifier, the datasets(scaling or not depends on classifier), 
        the path and name for saving the prediction, 
        train the model, predict both the training data and the test data,
        finally save them separately.
        ex. save_name = 'rf_5.csv'
    """
    assert '.csv' in save_name
    classifier.fit(X_train, y_train)
    # predict the training data and save it
    y_train_pred = classifier.predict(X_train)
    df = pd.DataFrame(y_train_pred, columns=['category'])
    df.index.name = 'id'
    df.to_csv(path + 'train_' + save_name, index=True, header=True)
    print ('f1 score of the training prediction : %f' % f1_score(y_train, y_train_pred, average='micro'))
    # predict the test data and save it
    y_test_pred = classifier.predict(X_test)
    df = pd.DataFrame(y_test_pred, columns=['category'])
    df.index.name = 'id'
    df.to_csv(path + 'test_' + save_name, index=True, header=True)


def LRTuning(c_range, tol_range, X, y, save_name, path):
    """
        Given ranges of parameter C and tol, the datasets(after scaling), 
        the path and name for saving the result,
        do cross validation and calculate f1 score for each value of C and tol, 
        and save the figure.
        ex. save_name = 'lr_1.png'
    """
    assert '.png' in save_name
    params = []
    scores = []
    for c in c_range:
        for t in tol_range:
            clf_lr = LogisticRegression(C=c, tol=t)
            params.append((c, t))
            scores.append(CV(clf_lr, X, y))
    print ([(params[i], scores[i]) for i in range(len(scores))])
    fig = plt.figure()
    x_range = range(len(scores))
    plt.plot(x_range, scores)
    plt.xticks(x_range, params)
    plt.xlabel('Value of (C, tol)')
    plt.ylabel('Cross validation f1 score')
    plt.title('LR Tuning')
    plt.grid(True)
    plt.legend()
    # plt.show()
    fig.savefig(path + save_name)


def SVMTuning(c_range, X, y, save_name, path):
    """
        Given a range of parameter C, the datasets(after scaling), 
        the path and name for saving the result,
        do cross validation and calculate f1 score for each value of C, 
        and save the figure.
        ex. save_name = 'svc_1.png'
    """
    assert '.png' in save_name
    scores = []
    for c in c_range:
        clf_svc = LinearSVC(C=c)
        scores.append(CV(clf_svc, X, y))
    print ([(c_range[i], scores[i]) for i in range(len(scores))])
    fig = plt.figure()
    plt.plot(c_range, scores)
    plt.xlabel('Value of C')
    plt.ylabel('Cross validation f1 score')
    plt.title('Linear SVC Tuning')
    plt.grid(True)
    plt.legend()
    # plt.show()
    fig.savefig(path + save_name)


def RFTuning(param_range, X, y, save_name, path):
    """
        Given a set of parameters, the datasets(no need of scaling), 
        the path and name for saving the result,
        do cross validation and calculate f1 score 
        for each combination of parameters, and save the figure.
        ex. save_name = 'rf_1.png'
            param_range = {'n_estimators':[50,100,200,400], 
                        'max_depth':[3,4], 'min_samples_split':[50,200,400]}
    """
    assert '.png' in save_name
    params = []
    scores = []
    for n in param_range['n_estimators']:
        for dep in param_range['max_depth']:
            for nsplit in param_range['min_samples_split']:
                clf_rf = RandomForestClassifier(n_estimators=n, max_depth=dep, min_samples_split=nsplit, random_state=0)
                params.append((n, dep, nsplit))
                scores.append(CV(clf_rf, X, y))
    print ([(params[i], scores[i]) for i in range(len(scores))])
    fig = plt.figure()
    x_range = range(len(scores))
    plt.plot(x_range, scores)
    plt.xticks(x_range, params)
    plt.xlabel('Value of (n_estimators, max_depth, min_samples_split)')
    plt.ylabel('Cross validation f1 score')
    plt.title('RF Tuning')
    plt.grid(True)
    plt.legend()
    # plt.show()
    fig.savefig(path + save_name)


def RFRegTrainAndPredict(rf_reg, X_train, y_train, X_test, save_name, path):
    """
        Given a defined random forest regressor, the datasets(no need of scaling), 
        the path and name for saving the result,
        train the model, predict both the training data and the test data(probabilities),
        choose the threshold which gives the best f1 score on training set,
        transform the prediction to 0/1 and finally save them separately.  
        ex. save_name = 'rfreg_1.csv'
            rf_reg = RandomForestRegressor(n_estimators=100, 
                                        max_depth=5, min_samples_split=200)
    """
    assert '.csv' in save_name
    # train the model and predict the probabilities
    rf_reg.fit(X_train, y_train)
    y_train_pred = rf_reg.predict(X_train)
    y_test_pred = rf_reg.predict(X_test)
    # choose the best f1 score on training set
    reg_train = np.zeros((len(y_train), 2))
    reg_train[:, 0] = np.array(y_train)
    reg_train[:, 1] = y_train_pred
    indice = np.argsort(reg_train[:,1])[::-1]
    reg_train = reg_train[indice]
    p_best, r_best, f1_best, ts = 0, 0, 0, 0
    num_ones = sum(y_train)
    tp = 0
    for idx, row in enumerate(reg_train):
        if row[0] == 1:
            tp += 1
        p = tp / (idx + 1)
        r = tp / num_ones
        f1 = 2 * p * r / (p + r)
        if f1 > f1_best:
            p_best, r_best, f1_best = p, r, f1
            ts = row[1]
    print (p_best, r_best, f1_best, ts)
    # transform the prediction of training set and save it
    y_train_pred = y_train_pred > ts
    y_train_pred = y_train_pred.astype(int)
    df = pd.DataFrame(y_train_pred, columns=['category'])
    df.index.name = 'id'
    df.to_csv(path + 'train_' + save_name, index=True, header=True)
    print ('f1 score of the training prediction : %f' % f1_score(y_train, y_train_pred, average='micro'))
    # transform the prediction of test set and save it
    y_test_pred = y_test_pred > ts
    y_test_pred = y_test_pred.astype(int)
    df = pd.DataFrame(y_test_pred, columns=['category'])
    df.index.name = 'id'
    df.to_csv(path + 'test_' + save_name, index=True, header=True)


def XGBTuning(param_range, X, y, save_name, path):
    """
        Given a set of parameters, the datasets(no need of scaling), 
        the path and name for saving the result,
        do cross validation and calculate f1 score for each combination of parameters, 
        and save the figure.
        ex. save_name = 'xgb_1.png'
            param_range = {'n_estimators':[50,100,200,400], 'max_depth':[2,3,4], 
            'learning_rate':[0.1,0.2,0.3], 'sub':[0]}
    """
    assert '.png' in save_name
    params = []
    scores = []
    for n in param_range['n_estimators']:
        for dep in param_range['max_depth']:
            for lr in param_range['learning_rate']:
                for sub in param_range['sub']:
                    clf_xgb = XGBClassifier(n_estimators=n, max_depth=dep, 
                                            booster='gbtree', objective="binary:logistic", 
                                            learning_rate=lr, subsample=sub, 
                                            colsample_bytree=sub, colsample_bylevel=sub, 
                                            random_state=0)
                    params.append((n, dep, lr, sub))
                    scores.append(CV(clf_xgb, X, y))
    print ([(params[i], scores[i]) for i in range(len(scores))])
    fig = plt.figure()
    x_range = range(len(scores))
    plt.plot(x_range, scores)
    plt.xticks(x_range, params)
    plt.xlabel('Value of (n_estimators, max_depth, learning_rate, subsample)')
    plt.ylabel('Cross validation f1 score')
    plt.title('XGB Tuning')
    plt.grid(True)
    plt.legend()
    # plt.show()
    fig.savefig(path + save_name)


def XGBTrainAndPredict(clf_xgb, esrounds, X, y, X_test, save_name, path):
    """
        Given a defined xgbclassifier, the datasets(no need of scaling), 
        the path and name for saving the result,
        train the model, predict both the training data and the test data,
        and finally save them separately.  
        ex. save_name = 'xgb_1.csv'
            clf_xgb = XGBClassifier(n_estimators=300, max_depth=2, 
                                    booster='gbtree', objective="binary:logistic", 
                                    learning_rate=.1, subsample=.2, 
                                    colsample_bytree=.2, colsample_bylevel=.2, 
                                    random_state=0)
    """
    assert '.csv' in save_name
    # shuffle and split training and validation sets, train the model
    X, y = shuffle(X, y, random_state=0)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, 
                                                        random_state=0, shuffle=True)
    clf_xgb.fit(X_train, y_train, eval_metric='logloss', eval_set=[(X_valid,y_valid)], early_stopping_rounds=esrounds)
    # predict the training and validation data and save it
    y_pred = clf_xgb.predict(X)
    df = pd.DataFrame(y_pred, columns=['category'])
    df.index.name = 'id'
    df.to_csv(path + 'train_' + save_name, index=True, header=True)
    print ('f1 score of the training prediction : %f' % f1_score(y, y_pred, average='micro'))
    # predict the test data and save it
    y_test_pred = clf_xgb.predict(X_test)
    df = pd.DataFrame(y_test_pred, columns=['category'])
    df.index.name = 'id'
    df.to_csv(path + 'test_' + save_name, index=True, header=True)


def MLPTuning(hidden_layers, alpha_range, X, y, save_name, path):
    """
        Given the structure, the range of parameter alpha, 
        the datasets(after scaling), the path and name for saving the result,
        do cross validation and calculate f1 score for each value of alpha, 
        and save the figure.
        ex. save_name = 'mlp_1.png'
            hidden_layers = (30, 10, 5)
            alpha_range = [1e-4, 1e-3, 1e-2, 1e-1]
    """
    assert '.png' in save_name
    scores = []
    for a in alpha_range:
        clf = MLPClassifier(hidden_layer_sizes=hidden_layers, 
                            alpha=a, 
                            max_iter=500, 
                            tol=1e-7,
                            verbose=True)
        scores.append(CV(clf, X, y))
    print ([(alpha_range[i], scores[i]) for i in range(len(scores))])
    fig = plt.figure()
    plt.xscale('log')
    plt.plot(alpha_range, scores)
    plt.xlabel('Value of alpha')
    plt.ylabel('Cross validation f1 score')
    plt.title('MLP Tuning')
    plt.grid(True)
    plt.legend()
    # plt.show()
    fig.savefig(path + save_name)


