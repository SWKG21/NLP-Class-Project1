import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models import *

train = pd.read_csv('../../project1/data/train_all2.csv')
X_train = train.iloc[:, 5:]
X_train.drop(labels=['pa'], axis=1, inplace=True)
y_train = train.loc[:,'link']
test = pd.read_csv('../../project1/data/test_all2.csv')
X_test = test.iloc[:, 4:]
X_test.drop(labels=['pa'], axis=1, inplace=True)

path = '../../project1/result/'

param_range = {'n_estimators':[100,200,300,400,500], 'max_depth':[5], 'min_samples_split':[200]}
RFTuning(param_range, X_train, y_train, 'rf1.png', path)
param_range = {'n_estimators':[100,200,300,400,500], 'max_depth':[2], 
                'learning_rate':[0.1], 'sub':[0.1]}
XGBTuning(param_range, X_train, y_train, 'xgb1.png', path)

# X_train, X_test = scaling(X_train, X_test, Standard=True)

# LRTuning([1e-1,1e3,1e7], [1e-5], X_train, y_train, 'lr1.png', path)
# SVMTuning([1e-1,1e3,1e7], X_train, y_train, 'svm1.png', path)