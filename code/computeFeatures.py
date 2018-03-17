import numpy as np
import pandas as pd
from utils import *

""" Run this script will compute some features and save the result."""

# load data
node_info = pd.read_csv('../../project1/data/node_information.csv', header=None)
node_info.columns = ['id', 'year', 'title', 'authors', 'journal', 'abstract']
train = pd.read_csv('../../project1/data/train_treated.csv')
train = train.iloc[:5, :5]
# compute 4 features
train = computeYearDifference(node_info, train)
train = computeCommonAuthors(node_info, train)
train = computeTitleOverlap(node_info, train)
train = computeJournalOverlap(node_info, train)
# compute abstract centroid similarity and wmdistance
texts = [tokenizer(node_info.loc[i, 'abstract']) for i in range(len(node_info))]
texts = filterByFrequency(texts)
w2v = buildModel(texts)
centroids = computeCentroids(texts, w2v)
train['centroid_sim'] = train.apply(lambda row: computeCentroidSim(centroids, row['rno1'], row['rno2']), axis=1)
train['wmd'] = train.apply(lambda row: computeWMD(texts, row['rno1'], row['rno2'], w2v), axis=1)
# output
train.to_csv('train_tmp.csv', index=False)

