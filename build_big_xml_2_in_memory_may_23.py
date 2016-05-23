#%%
import collections
import json
import nltk
import numpy
import gensim
import os
import itertools
import random
import re
import time
import matplotlib.pyplot as plt
import pattern.en, pattern.search
import enchant
import sklearn.cluster
import scipy.cluster.hierarchy
import stemming.porter2
from miningTools import stemSentence
import sys

import sqlalchemy, sqlalchemy.orm

import mldb2

session, engine = mldb2.getSession('/home/bbales2/scraping/webpage/db/sentences.document.sqlite.mldb2.db')
#%%
stemmedStops = stemSentence(' '.join(nltk.corpus.stopwords.words("english")))
#%%
class LoaderStopwords(object):
    def __init__(self, clss):
        self.clss = clss

    def __iter__(self):
        notword = re.compile(r'[^\w -]')
        for i, sentence in enumerate(itertools.chain(*[session.query(cls).order_by(cls.id).yield_per(100) for cls in self.clss])):
            stemmed = stemSentence(sentence.string)
            filtered = [word for word in stemmed if word not in stemmedStops and len(notword.sub('', word)) > 0 and len(word) < 75]
            print i
            yield filtered
#%%
#sentence_classes = [mldb2.AbstractSentence]
    #[mldb2.BodySentence, mldb2.AbstractSentence, mldb2.FigureSentence]
sentence_classes = [mldb2.AbstractSentence]
index2id = list(itertools.chain(*[[a[0] for a in session.query(cls).order_by(cls.id).with_entities(cls.id)] for cls in sentence_classes]))

prefix = 'body_abstract_figure.xml'

with open('/home/bbales2/scraping/webpage/db/{0}.index2id.big'.format(prefix), 'w') as f:
    f.write(json.dumps(index2id))

notword = re.compile(r'[^\w -]')
#%%
for cls in sentence_classes:
    print session.query(cls).count()#order_by(cls.id).yield_per(1000)
#%%
tmp1 = time.time()
loader_stopwords = []
word2vec_loader = []
for i, sentence in enumerate(itertools.chain(*[session.query(cls).order_by(cls.id).yield_per(1000) for cls in sentence_classes])):
    stemmed = stemSentence(sentence.string)
    word2vec_loader.append([word for word in stemmed if len(notword.sub('', word)) > 0 and len(word) < 75])
    filtered = [word for word in stemmed if word not in stemmedStops and len(notword.sub('', word)) > 0 and len(word) < 75]
    loader_stopwords.append(filtered)

    #if i > 500:
    #    break
    print i
print "Loading words time: ", time.time() - tmp1
#%%
tmp = time.time()
dictionary = gensim.corpora.dictionary.Dictionary(loader_stopwords)
dictionary.filter_extremes(no_below = 5, no_above = 0.5)
dictionary.compactify()
print 'Dictionary time: ', time.time() - tmp
sys.stdout.flush()

dictionary.save('/home/bbales2/scraping/webpage/db/{0}.dictionary.big.2'.format(prefix))

bow_loader = []
#dictionary = gensim.corpora.dictionary.Dictionary.load('/home/bbales2/scraping/webpage/db/{0}.dictionary.big'.format(prefix))
for sentence in loader_stopwords:
    bow_loader.append(dictionary.doc2bow(sentence))

import time
tmp = time.time()
tfidf = gensim.models.TfidfModel(bow_loader)
print 'Tfidf time: ', time.time() - tmp
sys.stdout.flush()
tfidf.save('/home/bbales2/scraping/webpage/db/{0}.tfidf.big.2'.format(prefix))

tfidf_loader = []
for sentence in bow_loader:
    tfidf_loader.append(tfidf[sentence])

tmp = time.time()
numberTopics = 300
lsi = gensim.models.lsimodel.LsiModel(corpus = tfidf_loader, num_topics = numberTopics, id2word = dictionary)
print 'LSI Time: ', time.time() - tmp
lsi.save('/home/bbales2/scraping/webpage/db/{0}.lsi.big.2'.format(prefix))
sys.stdout.flush()

lsi_loader = []
for sentence in tfidf_loader:
    lsi_loader.append(lsi[sentence])

tmp = time.time()
index = gensim.similarities.docsim.Similarity('/home/bbales2/scraping/lsi_index/{0}.index.big.2'.format(prefix), lsi_loader, num_features = numberTopics)
print 'Index Time: ', time.time() - tmp
sys.stdout.flush()
index.save('/home/bbales2/scraping/webpage/db/{0}.index.big.2'.format(prefix))
#%%
tmp = time.time()
word2vec = gensim.models.word2vec.Word2Vec(word2vec_loader, workers = 4)
word2vec.save('/home/bbales2/scraping/webpage/db/{0}.word2vec.big.2'.format(prefix))
print 'Word2Vec Training Time: ', time.time() - tmp
#%%