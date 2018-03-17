import random
import numpy as np
import igraph
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cosine
from collections import Counter
import nltk
import csv
import pandas as pd
from gensim.models.word2vec import Word2Vec
from scipy import sparse
from scipy.sparse.linalg import svds

#===================================== Data Loading =======================================

with open("training_set.txt", "r") as f:
    reader = csv.reader(f)
    training_set  = list(reader)

training_set = [element[0].split(" ") for element in training_set]

with open("testing_set.txt", "r") as f:
    reader = csv.reader(f)
    testing_set  = list(reader)

testing_set = [element[0].split(" ") for element in testing_set]

with open("node_information.csv", "r") as f:
    reader = csv.reader(f)
    node_info  = list(reader)

IDs = [element[0] for element in node_info]

#=========================== compute TFIDF vector of abstracts ============================

# corpus contains the abstracts of all the papers
corpus = [element[5] for element in node_info]
vectorizer = TfidfVectorizer(stop_words="english")
# features_TFIDF contains the original tfidf vectors, 
# each row is a node in the order of node_info
features_TFIDF = vectorizer.fit_transform(corpus)

# use svd to compute the tfidf vectors reduced to a dimension of 300
print('start svd')
sparse_TFIDF = sparse.csc_matrix(features_TFIDF)
u, s, vt = svds(sparse_TFIDF, k=300)
reduced_TFIDF = u.dot(np.diag(s))
print('svd completed')

#========================== word embedding for words in abstracts =========================

nltk.download('punkt') # for tokenization
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()
print('tokenizing abstracts')
cleaned_abstracts = []
# preprocessing for word embeddings
for idx, abstract in enumerate(corpus):
    tokens = abstract.split(' ')
    tokens = [token for token in tokens if token not in stpwds]
    # remove tokens shorter than 3 characters in size
    tokens = [token for token in tokens if len(token)>2]
    # remove tokens exceeding 25 characters in size
    tokens = [token for token in tokens if len(token)<=25]
    cleaned_abstracts.append(tokens)
    if idx % round(len(corpus)/10) == 0:
        print(idx)

# create empty word vectors for the words in vocabulary 
# we set size=300 to match dim of GNews word vectors
my_q = 300
mcount = 5
w2v = Word2Vec(size=my_q, min_count=mcount)
w2v.build_vocab(cleaned_abstracts)

# sanity check (last statement should return True)
vocab = w2v.wv.vocab.keys()
all_tokens = [token for sublist in cleaned_abstracts for token in sublist]
t_counts = dict(Counter(all_tokens))
if len(vocab) == len([token for token, count in t_counts.iteritems() if count>=mcount]):
    print('sanity check passed')
else:
    print('sanity check failed!')

path_to_google_news = '../course4/'
w2v.intersect_word2vec_format(path_to_google_news + 'GoogleNews-vectors-negative300.bin', binary=True)
print('word vectors loaded')

# NOTE: in-vocab words without an entry in the Google News file are not removed from the vocabulary
# instead, their vectors are silently initialized to random values
# we can detect those vectors via their norms which approach zero

print('removing no-entry words and infrequent words')
norms = [np.linalg.norm(w2v[word]) for word in vocab]
idxs_zero_norms = [idx for idx,norm in enumerate(norms) if norm<=0.05]
no_entry_words = [vocab[idx] for idx in idxs_zero_norms]

no_entry_words = set(no_entry_words)
for idx,doc in enumerate(cleaned_abstracts):
    cleaned_abstracts[idx] = [token for token in doc if token not in no_entry_words and t_counts[token]>=mcount]
    if idx % round(len(corpus)/10) == 0:
        print(idx)

#================================== Construct a graph =========================================

print('building graph.')

edges = [(element[0],element[1]) for element in training_set if element[2]=="1"]
nodes = IDs
# create empty directed graph
g = igraph.Graph(directed=True) 
# add vertices
g.add_vertices(nodes)
# add edges
g.add_edges(edges)

print('graph built.')

#=========================== Compute features for training examples ============================

# number of overlapping words in title
overlap_title = []
# temporal distance between the papers
temp_diff = []
# number of common authors
comm_auth = []
# tfidf cosine similarity
tfidf_cos_sim = []
# reduced tfidf cosine similarity
reduced_tfidf_sim = []
# number of common neighbors
comm_neighbors = []
# link-based Jaccard coefficient
jaccard_coeff = []
# difference in the number of in-links
in_diff = []
# number of times target article is cited
tgt_citation = []
# gaussian representation similarity
gaussian_sim = []
# similarity of journals
journal_sim = []

counter = 0
for i in xrange(len(training_set)):
    source = training_set[i][0]
    target = training_set[i][1]
    
    index_source = IDs.index(source)
    index_target = IDs.index(target)
    
    source_info = node_info[index_source]
    target_info = node_info[index_target]
    
    #------------------compute overlap_title-------------------

	# convert to lowercase and tokenize
    source_title = source_info[2].lower().split(" ")
	# remove stopwords
    source_title = [token for token in source_title if token not in stpwds]
    source_title = [stemmer.stem(token) for token in source_title]
    
    target_title = target_info[2].lower().split(" ")
    target_title = [token for token in target_title if token not in stpwds]
    target_title = [stemmer.stem(token) for token in target_title]
    overlap_title.append(len(set(source_title).intersection(set(target_title))))

    #----------------- compute temp_diff -----------------------

    temp_diff.append(int(source_info[1]) - int(target_info[1]))

    #----------------- compute comm_auth -----------------------

    source_auth = source_info[3].split(",")
    target_auth = target_info[3].split(",")

    comm_auth.append(len(set(source_auth).intersection(set(target_auth))))

    #----------------- compute comm_neighbors ------------------

    common_neighs = [v for v in g.neighbors(source) if v in g.neighbors(target)]

    comm_neighbors.append(len(common_neighs))
    
    #------------------ compute tfidf vectors ------------------

    tfidf_cos_sim.append(cosine(features_TFIDF[index_source], features_TFIDF[index_target])[0][0])
    reduced_tfidf_sim.append(cosine(reduced_TFIDF[index_source].reshape(1, -1), 
                                    reduced_TFIDF[index_target].reshape(1, -1))[0][0])
    
    #------------------ compute jaccard_coeff -------------------

    jaccard_coeff.append(g.similarity_jaccard(pairs=[(index_source, index_target)])[0])
 
    #-------------- compute the number of in-links --------------

    in_diff.append(g.degree(index_target, mode='IN') - g.degree(index_source, mode='IN'))

    #-----  compute number of times target article is cited -----

    tgt_citation.append(g.degree(index_target, mode='IN'))

    #------- compute gaussian similarity of the abstracts --------

    p1_embeddings = np.concatenate([w2v[token].reshape(1, -1) for token in cleaned_abstracts[index_source]])
    mu1 = np.mean(p1_embeddings, axis=0)
    p1 = p1_embeddings - mu1
    cov1 = p1.T.dot(p1) / len(p1)
    # print p1_embeddings.shape, mu1.shape, p1.shape, cov1.shape
    p2_embeddings = np.concatenate([w2v[token].reshape(1, -1) for token in cleaned_abstracts[index_target]])
    mu2 = np.mean(p2_embeddings, axis=0)
    p2 = p2_embeddings - mu2
    cov2 = p2.T.dot(p2) / len(p2)
    sim_mu = cosine(mu1.reshape(1, -1), mu2.reshape(1, -1))[0][0]
    sim_cov = np.sum(cov1 * cov2) / (np.linalg.norm(cov1) * np.linalg.norm(cov2))
    gaussian_sim.append((sim_cov + sim_mu) * 0.5)
   
    #--------- compute the similarity of journals ----------------

    source_jour = source_info[4].split(".")
    target_jour = target_info[4].split(".")
    source_jour = (source_jour[:-1] if source_jour[-1] == '' else source_jour)
    target_jour = (target_jour[:-1] if target_jour[-1] == '' else target_jour)

    if source_jour == [] or target_jour == []:
        journal_sim.append(-1)
    else:
        inter = set(source_jour).intersection(set(target_jour))
        union = set(source_jour).union(set(target_jour))
        journal_sim.append(len(inter) * 1.0 / len(union))

    counter += 1
    if counter % 1000 == True:
        print(counter, "training examples processsed")

#=========================== write the features into a csv file ============================

# convert labels into integers then into column array
labels = [int(element[2]) for element in training_set]
train_data = pd.DataFrame()
train_data['overlap_title'] = overlap_title
train_data['temp_diff'] = temp_diff
train_data['comm_auth'] = comm_auth
train_data['tfidf_cos_sim'] = tfidf_cos_sim
train_data['reduced_tfidf_sim'] = reduced_tfidf_sim
train_data['comm_neighbors'] = comm_neighbors
train_data['jaccard_coeff'] = jaccard_coeff
train_data['in_diff'] = in_diff
train_data['tgt_citation'] = tgt_citation
train_data['gaussian_sim'] = gaussian_sim
train_data['journal_sim'] = journal_sim
train_data['labels'] = labels
train_data.to_csv('train_features.csv', index=False)
print('training set saved to csv.')

#=========================== Compute features for testing examples ============================

overlap_title_test = []
temp_diff_test = []
comm_auth_test = []
tfidf_cos_sim_test = []
reduced_tfidf_sim_test = []
comm_neighbors_test = []
jaccard_coeff_test = []
in_diff_test = []
tgt_citation_test = []
gaussian_sim_test = []
journal_sim_test = []
   
counter = 0
for i in xrange(len(testing_set)):
    source = testing_set[i][0]
    target = testing_set[i][1]
    
    index_source = IDs.index(source)
    index_target = IDs.index(target)
    
    source_info = node_info[index_source]
    target_info = node_info[index_target]
    
    #---------------- compute overlap_title ---------------------

    source_title = source_info[2].lower().split(" ")
    source_title = [token for token in source_title if token not in stpwds]
    source_title = [stemmer.stem(token) for token in source_title]
    
    target_title = target_info[2].lower().split(" ")
    target_title = [token for token in target_title if token not in stpwds]
    target_title = [stemmer.stem(token) for token in target_title]
    
    overlap_title_test.append(len(set(source_title).intersection(set(target_title))))

    #-------------------- compute temp_diff -----------------------
    
    temp_diff_test.append(int(source_info[1]) - int(target_info[1]))

    #--------------------- compute comm_auth ----------------------

    source_auth = source_info[3].split(",")
    target_auth = target_info[3].split(",")
     
    comm_auth_test.append(len(set(source_auth).intersection(set(target_auth))))

    #-------------------- compute tfidf vectors --------------------

    tfidf_cos_sim_test.append(cosine(features_TFIDF[index_source], features_TFIDF[index_target])[0][0])
    reduced_tfidf_sim_test.append(cosine(reduced_TFIDF[index_source].reshape(1, -1), 
                                    reduced_TFIDF[index_target].reshape(1, -1))[0][0])

    #------------------ compute common neighbors --------------------

    common_neighs = [v for v in g.neighbors(source) if v in g.neighbors(target)]
    comm_neighbors_test.append(len(common_neighs))

    #------------------ compute jaccard_coeff ------------------------

    jaccard_coeff_test.append(g.similarity_jaccard(pairs=[(index_source, index_target)])[0])

    #------------------ compute the number of in_links ---------------

    in_diff_test.append(g.degree(index_target, mode='IN') - g.degree(index_source, mode='IN'))

    #------------------ compute target citation -------------------

    tgt_citation_test.append(g.degree(index_target, mode='IN'))

    #------------------ compute gaussian similarity -----------------
    p1_embeddings = np.concatenate([w2v[token].reshape(1, -1) for token in cleaned_abstracts[index_source]])
    mu1 = np.mean(p1_embeddings, axis=0)
    p1 = p1_embeddings - mu1
    cov1 = p1.T.dot(p1) / len(p1)
    p2_embeddings = np.concatenate([w2v[token].reshape(1, -1) for token in cleaned_abstracts[index_target]])
    mu2 = np.mean(p2_embeddings, axis=0)
    p2 = p2_embeddings - mu2
    cov2 = p2.T.dot(p2) / len(p2)
    sim_mu = cosine(mu1.reshape(1, -1), mu2.reshape(1, -1))[0][0]
    sim_cov = np.sum(cov1 * cov2) / (np.linalg.norm(cov1) * np.linalg.norm(cov2))
    gaussian_sim_test.append((sim_cov + sim_mu) * 0.5)
   
    #------------------ compute journal similarity -------------------
    source_jour = source_info[4].split(".")
    target_jour = target_info[4].split(".")
    source_jour = (source_jour[:-1] if source_jour[-1] == '' else source_jour)
    target_jour = (target_jour[:-1] if target_jour[-1] == '' else target_jour)

    if source_jour == [] or target_jour == []:
        journal_sim_test.append(-1)
    else:
        inter = set(source_jour).intersection(set(target_jour))
        union = set(source_jour).union(set(target_jour))
        print('len_union', len(union))
        journal_sim_test.append(len(inter) * 1.0 / len(union))

    counter += 1
    if counter % 1000 == True:
        print(counter, "testing examples processsed")
        
#========================== write the features into a csv file =======================

test_data = pd.DataFrame()
test_data['overlap_title'] = overlap_title_test
test_data['temp_diff'] = temp_diff_test
test_data['comm_auth'] = comm_auth_test
test_data['tfidf_cos_sim'] = tfidf_cos_sim_test
test_data['reduced_tfidf_sim'] = reduced_tfidf_sim_test
test_data['comm_neighbors'] = comm_neighbors_test
test_data['jaccard_coeff'] = jaccard_coeff_test
test_data['in_diff'] = in_diff_test
test_data['tgt_citation'] = tgt_citation_test
test_data['gaussian_sim'] = gaussian_sim_test
test_data['journal_sim'] = journal_sim_test
test_data.to_csv('test_features.csv', index=False)
print('test set saved to csv.')
