import numpy as np 
import pandas as pd 

from sklearn.preprocessing import MinMaxScaler, scale, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import  MLPClassifier
from sklearn.utils import shuffle
from sklearn.metrics import f1_score

path = '../../project1/data/'
train = pd.read_csv(path + 'train_all2.csv')
test = pd.read_csv(path + 'test_all2.csv')
# features = ['overlap_title', 'temp_diff', 'comm_auth', 'ada_ada_ind', 'reduced_tfidf_sim', 'comm_neighbors', 'in_diff', ']
#features = ['sim', 'year_diff', 'common_authors', 'cn', 'aai', 'title_overlap']
# X_train = train.drop(['id1', 'id2', 'jaccard_coeff', 'labels', 'ada_ada_ind', 'comm_auth', 'comm_neigh_s', 'title_sim'], axis=1).values
# y_train = train['labels']
# X_test = test.drop(['id1', 'id2', 'jaccard_coeff', 'comm_auth', 'ada_ada_ind', 'comm_neigh_s', 'title_sim'], axis=1).values
X_train = train.drop(['id1', 'id2', 'link', 'rno1', 'rno2', 'pa'], axis=1).values
y_train = train['link']
X_test = test.drop(['id1', 'id2', 'rno1', 'rno2', 'pa'], axis=1).values


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# X_train = scale(X_train)
# X_test = scale(X_test)
X_train, y_train = shuffle(X_train, y_train, random_state=0)

alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
for alpha in alphas:
    clf = MLPClassifier(hidden_layer_sizes=(30, 15, 5), 
                        alpha=alpha, 
                        max_iter=200, 
                        tol=1e-7,
                        verbose=False)
    # print ('cv', cross_val_score(clf, 
    #                            X_train, 
    #                            y_train, 
    #                            cv=5, 
    #                            scoring='f1', 
    #                            n_jobs=-1).mean())
    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    print ("training set f1 score is", f1_score(y_train, train_pred, average='micro'))

# test_pred = clf.predict(X_test)
# preds = pd.DataFrame()
# preds['id'] = range(len(test))
# preds['category'] = test_pred
# preds.to_csv(path+'../result/preds_nn_all.csv', index=False)
# print ('prediction saved')
