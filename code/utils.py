import numpy as np
import pandas as pd
from collections import defaultdict

from gensim.models.word2vec import Word2Vec
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
import nltk


def computeYearDifference(node_info, df):
    """
        Compute the year difference of two articles.

        Parameters
        ----------
            node_info: dataframe
                Dataframe containing node information.
            df: dataframe
                Dataframe containing the id of articles.
        
        Return
        ------
            Dataframe with computed year difference.
    """
    df['year1'] = df.apply(lambda row: node_info.loc[node_info['id']==row['id1'], 'year'].values[0], axis=1)
    df['year2'] = df.apply(lambda row: node_info.loc[node_info['id']==row['id2'], 'year'].values[0], axis=1)
    df['year_diff'] = df.apply(lambda row: row['year1'] - row['year2'], axis=1)
    df['year_diff'] = df['year_diff'].astype(int)
    df.drop(['year1', 'year2'], axis=1, inplace=True)
    return df


def processAuthor(authors):
    """
        Transform the raw string of authors to a set of authors' names.

        Parameters
        ----------
            authors: string
                Raw string of authors.
        
        Return
        ------
            Set with authors' names.
    """
    if not isinstance(authors, str):
        return set()
    names = authors.lower().split(',')
    names = [name.replace('.', ' ').split() for name in names]
    names = set([' '.join(name) for name in names])
    return names


def processTitle(title):
    """
        Transform the raw string of title to a set of words.

        Parameters
        ----------
            title: string
                Raw string of title.
        
        Return
        ------
            Set with title words.
    """
    stpwds = set(nltk.corpus.stopwords.words('english'))
    stemmer = PorterStemmer()
    if not isinstance(title, str):
        return set()
    tokens = title.lower().split()
    tokens = [token for token in tokens if token not in stpwds]
    tokens = [stemmer.stem(token) for token in tokens]
    return set(tokens)


def processJournal(journal):
    """
        Transform the raw string of journal to a set of words.

        Parameters
        ----------
            journal: string
                Raw string of journal.
        
        Return
        ------
            Set with journal words.
    """
    if not isinstance(journal, str):
        return set()
    return set(journal.lower().strip('."/,`()-\'').split('.'))


def computeBetweenTwo(node_info, id1, id2, feature_name):
        """
            Compute their common features
            and divide it with the average feature number of these two articles.

            Parameters
            ----------
                node_info: dataframe
                    Dataframe containing node information.
                id1: integer
                    id of the first article.
                id2: integer
                    id of the second article.
                feature_name: string
                    feature name to compute common number. ex. authors_split
            
            Return
            ------
                Computed common number of this feature.
        """
        f1 = node_info.loc[node_info['id']==id1, feature_name].values[0]
        f2 = node_info.loc[node_info['id']==id2, feature_name].values[0]
        avl = (len(f1) + len(f2)) / 2.0
        if avl == 0:
            return 0
        else:
            return len(f1 & f2) / float(avl)


def computeCommonAuthors(node_info, df):
    """
        Compute the number of common authors of two articles for all rows. 
        
        Parameters
        ----------
            node_info: dataframe
                Dataframe containing node information.
            df: dataframe
                Dataframe containing the id of articles.
        
        Return
        ------
            Dataframe with computed common authors.
    """
    node_info['authors_split'] = node_info.apply(lambda row: processAuthor(row['authors']), axis=1)
    df['common_authors'] = df.apply(lambda row: computeBetweenTwo(node_info, row['id1'], row['id2'], 'authors_split'), axis=1)
    return df


def computeTitleOverlap(node_info, df):
    """
        Compute the number of common words in titles of two articles for all rows.

        Parameters
        ----------
            node_info: dataframe
                Dataframe containing node information.
            df: dataframe
                Dataframe containing the id of articles.
        
        Return
        ------
            Dataframe with computed common title words.
    """
    node_info['title_split'] = node_info.apply(lambda row: processTitle(row['title']), axis=1)
    df['title_overlap'] = df.apply(lambda row: computeBetweenTwo(node_info, row['id1'], row['id2'], 'title_split'), axis=1)
    return df


def computeJournalOverlap(node_info, df):
    """
        Compute the number of common words in journal of two articles for all rows.

        Parameters
        ----------
            node_info: dataframe
                Dataframe containing node information.
            df: dataframe
                Dataframe containing the id of articles.
        
        Return
        ------
            Dataframe with computed common journal words.
    """
    node_info['journal_split'] = node_info.apply(lambda row: processJournal(row['journal']), axis=1)
    df['journal_overlap'] = df.apply(lambda row: computeBetweenTwo(node_info, row['id1'], row['id2'], 'journal_split'), axis=1)
    return df


def tokenizer(line, stem=False):
    """
        Tokenization for a long raw string.

        Parameters
        ----------
            line: string
                The long string to tokenizer.
            stem: boolean
                Whether use stemming or not.
        
        Return
        ------
            List of tokens.
    """
    # Remove space at the beginning and end. Split. Filter null string. 
    tokens = [token for token in line.strip().split() if len(token)>1 and token not in STOPWORDS]
    # Remove digit in tokens
    tokens = [''.join([elt for elt in token if not elt.isdigit()]) for token in tokens]
    if stem:
        # stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    # Filter null string after stemming and remove stopwords
    tokens = [token for token in tokens if len(token)>1 and token not in STOPWORDS]
    return tokens


def filterByFrequency(texts, threshold=2):
    """
        Remove tokens with frequency less than a threshold.

        Parameters
        ----------
            texts: list of lists of tokenized strings
                Texts to filter.
            threshold: integer
                Threshold to filter tokens.

        Return
        ------
            List of lists of tokens.
    """
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token]>threshold] for text in texts]
    return texts


def buildModel(texts, vdim=300, mcount=1):
    """
        Build a word2vec model.

        Parameters
        ----------
            texts: list of lists of tokenized and filtered strings
                Texts to build the model.
            vdim: integer
                Vector dimension.
            mcount: integer
                Minimum frequency of word to build the model.
        
        Return
        ------
            Word2vec model object
    """
    w2v = Word2Vec(size=vdim, min_count=mcount)
    w2v.build_vocab(texts)
    w2v.intersect_word2vec_format('../../GoogleNews-vectors-negative300.bin', binary=True)
    return w2v


def computeCentroids(texts, w2v_model):
    """
        Compute a centroid vector for each text.

        Parameters
        ----------
            texts: list of lists of tokenized and filtered strings
                Texts to compute centroids.
            w2v_model: word2vec model object
                Model used to generate the word vectors. 
        
        Return
        ------
            Centroid vectors for texts.
    """
    vdim = w2v_model.layer1_size
    centroids = np.empty(shape=(len(texts), vdim))
    for idx,doc in enumerate(texts):
        centroid = np.mean(np.concatenate([w2v_model[token].reshape(1,-1) for token in doc]), axis=0)
        centroids[idx,:] = centroid
    return centroids


def computeCentroidSim(centroids, rno1, rno2):
    """
        Compute the centroid similarity for two texts.

        Parameters
        ----------
            centroids: array
                Centroid vectors for tokenized and filtered texts.
            rno1: integer
                Row number of the first text.
            rno2: integer
                Row number of the second text.
        
        Return
        ------
            Centroid similarity for two texts.
    """
    rno1 = int(rno1)
    rno2 = int(rno2)  
    c1 = centroids[rno1:(rno1+1),:]
    c2 = centroids[rno2:(rno2+1),:]
    return cosine_similarity(c1, c2)[0][0]


def computeWMD(texts, rno1, rno2, w2v_model):
    """
        Compute the word mover distance for two texts.

        Parameters
        ----------
            texts: list of lists of strings
                Texts to compute wmdistance.
            rno1: integer
                Row number of the first text.
            rno2: integer
                Row number of the second text.
            w2v_model: word2vec model object
                Model used to generate the word vectors.
        
        Return
        ------
            Word Mover Distance for two texts.
    """
    rno1 = int(rno1)
    rno2 = int(rno2)
    sent1 = texts[rno1]
    sent2 = texts[rno2]
    return w2v_model.wv.wmdistance(sent1, sent2)
