import numpy as np
from scipy import sparse
import pandas as pd
import networkx as nx

##### load training and testing dataset
train = pd.read_csv('data/training_set.txt', delimiter=' ', header=None)
train.columns = ['id1', 'id2', 'link']
test = pd.read_csv('data/testing_set.txt', delimiter=' ', header=None)
test.columns = ['id1', 'id2']

##### extract all known links in the graph
links = train[train['link']==1]
edgelist = links[['id1', 'id2']].values
edgelist = [tuple(e) for e in edgelist] # adapt edgelist to the format accepted by networkx graph constrctor

##### construct the graph from know edges
G = nx.Graph(edgelist)

##### map node id to its order in nodes list
nodes = G.nodes()
nodeid2rno = {node:idx for idx, node in enumerate(nodes)}
rno2nodeid = {idx:node for idx, node in enumerate(nodes)}
# these two dictionary allows to easily retrieve the row vector of a specific node by its id

##### get the ajdacency matrix, represented in sparse format for sake of saving memory space
adjacency_matrix = sparse.csc_matrix(nx.adjacency_matrix(G).todense())

##### get degrees of nodes and transform to a one dimension array
degrees = G.degree()
degrees = np.array([d[1] for d in degrees])

##### get all pairs of nodes listed in training set and testing set
train_pairs = train[['id1', 'id2']].values
train_pairs = [tuple(e) for e in train_pairs]
test_pairs = test[['id1', 'id2']].values
test_pairs = [tuple(e) for e in test_pairs]

##### calculate three values: jaccard coefficient, resource allocation index and adamic adar index
methods = {'jc': nx.jaccard_coefficient,
           'rai': nx.resource_allocation_index,
           'aai': nx.adamic_adar_index}
for m_name, method in methods.items():
    train_preds = method(G, train_pairs)
    train_scores = np.zeros(len(train))
    # extract the predicted value from the return
    for i, (_, _, p) in enumerate(train_preds):
        train_scores[i] = p
    # attach this value as a new column to training dataset
    train[m_name] = pd.Series(train_scores, index=train.index)
    
    test_preds = method(G, test_pairs)
    test_scores = np.zeros(len(test))
    # extract the predicted value from the return 
    for i, (_, _, p) in enumerate(test_preds):
        test_scores[i] = p
    # attach this value as a new column to testing dataset
    test[m_name] = pd.Series(test_scores, index=test.index)

##### calculate number of common neighbours
train['cn'] = train.apply(lambda row: len(sorted(nx.common_neighbors(G, row['id1'], row['id2']))), axis=1)
test['cn'] = test.apply(lambda row: len(sorted(nx.common_neighbors(G, row['id1'], row['id2']))), axis=1)

#========== define some useful functions ==========#
def predictRow(row, m_name, ts, default_label):
    """This function predict if link exists by comparing the value of m_name to the corresponding threshold,
       a default label is passed as argument in case that the value is equal to the threshold.
       The default label is the major label of all training instance with value equal to the threshold.
    """
    if row[m_name] > ts:
        return 1
    elif row[m_name] == threshold:
        return default_label
    else:
        return 0

def findThreshold(label, score):
    """This function searches for the best threshold of score to optimize F-score."""
    idx_sort = np.argsort(score)[::-1]
    score_sorted = score[idx_sort]
    label_sorted = score[idx_sort]

    # run through the lists to optimize f1 score
    p_best, r_best, f1_best = 0, 0, 0
    threshold = 0
    num_ones = sum(label)
    tp = 0
    for cnt, (l, s) in enumerate(zip(label_sorted, score_sorted)):
        if l == 1:
            tp += 1
        p = tp / (cnt +1)
        r = tp / num_ones
        f1 = 2 * p * r / (p + r)
        if f1 > f1_best:
            p_best, r_best, f1_best = p, r, f1
            threshold = s
    
    # count the majority label at the threshold
    at_threshold = label_sorted[score_sorted == threshold]
    default_label = int(at_threshold.mean() > 0.5)

    return threshold, default_label

#==================================================#

##### predict links according to each index value
methods = ['jc', 'rai', 'aai', 'cn']

for m_name in methods:
    link = train.loc[:, 'link'].values
    score = train.loc[:, m_name].values
    # get threshold and default label for instances with value equal to the threshold
    threshold, default_label = findThreshold(link, score)
    
    # predict and stack prediction as a new column
    train['pred_'+m_name] = train.apply(lambda row: predictRow(row, m_name, threshold, default_label), axis=1)
    test['pred_'+m_name] = test.apply(lambda row: predictRow(row, m_name, threshold, default_label), axis=1)

##### extract prediction result of each method and save as a submission file
test.index.name = 'id'
for m_name in methods.keys():
    sub = test[['pred_'+m_name]]
    sub.columns = ['category']
    sub.to_csv('data/sub_{}.csv'.format(m_name))
    
##############################################################################
## following is the functions to calculate Random Walk with Restart and ######
## Superposed Random Walk scores. However, the computation needs massive #####
## matrix calculation. We didn't manage to finish them on our labtops. #######
##############################################################################

P = sparse.csc_matrix((adjacency_matrix / degrees).T)

def RWR(P, c=0.5):
    Q = sparse.csc_matrix(sparse.linalg.inv((sparse.eye(P.shape[0]) - c * P))) * (1 - c)
    return Q + Q.T


def SRW(P, q, t=0):
    q_ = np.array(q).reshape((len(q), 1))
    pie = np.eye(P.shape[0])
    tmp = np.array(pie) * q_
    srw = tmp + tmp.T
    for i in range(t):
        pie = pie.dot(P)
        tmp = np.array(pie) * q_
        srw += tmp + tmp.T
    return srw

srw = SRW(P, degrees, t=0)
