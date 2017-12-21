
from nltk.corpus import stopwords 
import string
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from math import *
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import PorterStemmer, WordPunctTokenizer
import sys
import string
from collections import OrderedDict

import re
import itertools
from collections import OrderedDict
from functools import cmp_to_key
import itertools
from nltk import tokenize
import math


# The minimum similarity for sentences to be considered similar by LexPageRank.
MIN_LEXPAGERANK_SIM = 0.2
EPSILON = 0.0001

# The maximum similarity between two sentences that one should be
# considered a duplicate of the other.
MAX_SIM_CUTOFF = 0.4

def cosine_similarity_metric3(a1, a2):
    return cosine_similarity(a1.reshape(1,-1), a2.reshape(1,-1))[0][0]

def clean(doc):
    """ Lemmatizing and Tokenization for text data/ doc. Arg: doc -> string"""
    exclude = set(string.punctuation) 
    lmtzr = WordNetLemmatizer()
    # tokens = WordPunctTokenizer().tokenize(doc)
    # clean = [token.lower() for token in tokens if token.lower()]
    tokenise = [i for i in doc.lower().split() ]

    punc_free = []

    for word in tokenise:
        if re.match("^\d+?\.\d+?$", word) is not None:
            punc_free.append(word)
        else:
            tmp = ''.join(ch for ch in word if ch not in exclude)
            punc_free.append(tmp)

    # punc_free = ''.join(ch for ch in tokenize if ch not in exclude)
    final = [lmtzr.lemmatize(word, pos = 'v') for word in punc_free]
    final_doc = " ".join(final)
    # for s in final:
    #   final_doc += s + " "
    return final_doc

# feature_names = []
#confirm if stopwrods need to be removed?
vectorizer = TfidfVectorizer(min_df=1, stop_words = 'english')
def tf_model(docs_clean,upto):
    vectorizer.fit(docs_clean[:upto])
    # feature_names = vectorizer.get_feature_names()

def gen_tfidf_vector_sentence(doc):
    # feature_names = []
    # #confirm if stopwrods need to be removed?
    # vectorizer = TfidfVectorizer(min_df=1, stop_words = 'english')
    # vectorizer.fit(cleaned_doc_list)
    # feature_names = vectorizer.get_feature_names()
    Y = vectorizer.transform([doc])
    Y = Y.toarray()
    return Y

# def is_repeat(sent, sents, vect_fun=tfidf_vectorize, max_sim=MAX_SIM_CUTOFF):
def is_repeat(sent, sents, max_sim=MAX_SIM_CUTOFF):
    # TODO: Incorporate synonyms to better discern similarity
    for other_sent in sents:
        x = gen_tfidf_vector_sentence( sent)
        y = gen_tfidf_vector_sentence(other_sent)
        if cosine_similarity_metric3(x, y) > max_sim:
            return True
    return False

def sim_adj_matrix(sents, min_sim=MIN_LEXPAGERANK_SIM):
    """Compute the adjacency matrix of a list of tokenized sentences,
    with an edjge if the sentences are above a given similarity."""
    # return [[1 if cosine_sim(s1, s2, tfidf_vectorize) > min_sim else 0
    
    graph = []
    for s1 in sents:
        graph_row = []
        for s2 in sents:
            x = gen_tfidf_vector_sentence(s1)
            y = gen_tfidf_vector_sentence(s2)
            if(cosine_similarity_metric3(x,y) > min_sim):
                graph_row.append(1)
            else:
                graph_row.append(0)
        graph.append(graph_row)
    # print("here2")
    return graph


def normalize_matrix(matrix):
    """Given a matrix of number values, normalize them so that a row
    sums to 1."""
    for i, row in enumerate(matrix):
        tot = float(sum(row))
        try:
            matrix[i] = [x / tot for x in row]
        except ZeroDivisionError:
            pass
    return matrix


def pagerank(matrix, d=0.85):
    n = len(matrix)
    rank = [1.0 / n] * n
    new_rank = [0.0] * n
    while not has_converged(rank, new_rank):
        rank = new_rank
        new_rank = [(((1.0-d) / n) +
                     d * sum((rank[i] * link) for i, link in enumerate(row)))
                    for row in matrix]
    return rank


def has_converged(x, y, epsilon=EPSILON):
    """Are all the elements in x are within epsilon of their y's?"""
    for a, b in zip(x, y):
        if abs(a - b) > epsilon:
            return False
    return True

def gen_summary_from_rankings(score, clean_sents, tok_sents, orig_sents, max_words):

    ranked_sents = sorted(zip(score, tok_sents, orig_sents, clean_sents), reverse=True)
    # summary, tok_summary = [], []
    summary, tok_summary, clean_summary = [], [],[]

    word_count = 0

    for score, tok_sent, orig_sent, clean_sent in ranked_sents:
        if word_count >= max_words:
            break
        # if (is_valid_sent_len(tok_sent) and
        if( not is_repeat(clean_sent, clean_summary)):
            summary.append(orig_sent)
            # tok_summary.append(tok_sent)
            clean_summary.append(clean_sent)
            word_count += len(tok_sent)

    return summary 

def gen_lexrank_summary(orig_sents, max_words,docs_clean,upto):
    tf_model(docs_clean,upto)
    tok_sents = [clean(orig_sent).split()
                 for orig_sent in orig_sents]
    clean_sents = [clean(orig_sent)
                 for orig_sent in orig_sents]
    # print("here")
    adj_matrix = normalize_matrix(sim_adj_matrix(clean_sents))
    # print("passed")
    rank = pagerank(adj_matrix)
    # return gen_summary_from_rankings(rank, tok_sents, orig_sents, max_words)
    return gen_summary_from_rankings(rank, clean_sents, tok_sents,orig_sents, max_words)
