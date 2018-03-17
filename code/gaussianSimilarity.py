import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity as cosine
from gensim.models.word2vec import Word2Vec
from gensim.parsing.preprocessing import STOPWORDS as stpwds
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

##### load node information
node_info = pd.read_csv('data/node_information.csv', header=None)
node_info.columns = ['id', 'year', 'title', 'authors', 'journal', 'abstract']


##### tool functions for text preprocessing
punct = string.punctuation.replace('-', '').replace("'", '')
my_reg = re.compile(r"(\b[-']\b)|[\W_]")

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

def my_vector_getter(word, wv):
    try:
        word_array = wv[word].reshape(1, -1)
        return (word_array)
    except KeyError:
        print('word: <{}> not in vocabulary!'.format(word))
        
def clean_string(string, punct=punct, my_regex=my_reg, to_lower=False):
    if to_lower:
        string = string.lower()
    str = re.sub('\s+', ' ', string)
    str = ''.join(l for l in str if l not in punct)
    str = my_regex.sub(lambda x: (x.group(1) if x.group(1) else ' '), str)
    str = re.sub(' +', ' ', str)
    str = str.strip()
    return str

##### pipeline of text preprocessing
def preprocessText(text):
    """This function preprocesses text: tokenize, remove digits, eliminate short (<2) words, stemmer"""
    doc = clean_string(text, punct, my_reg)
    tokens = doc.split()
    tokens = [token for token in tokens if token not in stpwds]
    tokens = [''.join([elt for elt in token if not elt.isdigit()]) for token in tokens]
    tokens = [token for token in tokens if len(token)>2]
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

def checkWordInVocab(word, wv):
    """This function checks if a word is known by a word2vec model, in order to handle key error exceptions"""
    try:
        wv[word]
        return True
    except KeyError:
        return False

##### extract abstract texts and preprocess them
texts = node_info['abstract']
texts = texts.apply(lambda doc: preprocessText(doc))

##### use GoogleNews word2vec embedding
my_q = 300
mcount = 5
w2v = Word2Vec(size=my_q, min_count=mcount)
w2v.build_vocab(texts)
w2v.intersect_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)

##### eliminate tokens which are not includen in the w2v model
texts = texts.apply(lambda doc: [token for token in doc if checkWordInVocab(token, w2v)])

##### define some tool functions #############################################
def computeMeanVector(row):
    """This functions computes the mean vector of all word vectors."""
    vectors = [w2v[token] for token in row['tokens']]
    vectors = np.array(vectors)
    rst = vectors.mean(axis=0)
    return list(rst)

def computeCovariance(row):
    """This functions computes the covariance matrix of token vectors."""
    vectors = [w2v[token] for token in row['tokens']]
    vectors = np.array(vectors)
    vectors -= np.array(row['mean_vector'])
    num_tokens = vectors.shape[1]
    return list(np.dot(vectors.T, vectors) / num_tokens)

def computeCentroidSim(rn1, rn2):
    """This function computes centroid similarity with mean vectors."""
    vec1 = np.array(abstracts.loc[rn1, 'mean_vector']).reshape(1, -1)
    vec2 = np.array(abstracts.loc[rn2, 'mean_vector']).reshape(1, -1)
    return cosine(vec1, vec2)[0, 0]

def computeCovSim(rn1, rn2):
    """This function computes covariance similarity with covariance matrices"""
    cov1 = np.array(abstracts.loc[rn1, 'cov'])
    cov2 = np.array(abstracts.loc[rn2, 'cov'])
    return sum(sum(cov1 * cov2)) / (np.linalg.norm(cov1) * np.linalg.norm(cov2))

##### compute centroid mean vector and covariance matrices of each row
abstracts = pd.DataFrame()
abstracts['tokens'] = texts
abstracts['mean_vector'] = abstracts.apply(lambda row: computeMeanVector(row), axis=1)
abstracts['cov'] = abstracts.apply(lambda row: computeCovariance(row), axis=1)

##### build a map between node_info id and row number for later use
id2rn = {i:rn for rn, i in enumerate(node_info['id'])}

##### keep only the mean_vector and cov columns and remove the token column to save memory
abstracts = abstracts[['mean_vector', 'cov']]

##### load training and testing data
train_treated = pd.read_csv('train_treated.csv')
test_treated = pd.read_csv('test_treated.csv')

##### compute centroid similarity
train_treated['centroid_sim'] = train_treated.apply(lambda row: computeCentroidSim(id2rn[row['id1']], id2rn[row['id2']]), axis=1)
test_treated['centroid_sim'] = test_treated.apply(lambda row: computeCentroidSim(id2rn[row['id1']], id2rn[row['id2']]), axis=1)

##### remove mean_vector column to save memory
abstracts = abstracts[['cov']]

##### compute covariance similarity
train_treated['cov_sim'] = train_treated.apply(lambda row: computeCovSim(id2rn[row['id1']], id2rn[row['id2']]), axis=1)
test_treated['cov_sim'] = test_treated.apply(lambda row: computeCovSim(id2rn[row['id1']], id2rn[row['id2']]), axis=1)

##### save to local disk
train_treated.to_csv('train_treated.csv', index=False)
test_treated.to_csv('test_treated.csv', index=False)
