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

class BOWLoader(object):
    def __init__(self, clss):
        self.clss = clss

    def __iter__(self):
        dictionary = gensim.corpora.dictionary.Dictionary.load('/home/bbales2/scraping/webpage/db/{0}.dictionary.big'.format(prefix))
        for sentence in LoaderStopwords(self.clss):
            yield dictionary.doc2bow(sentence)

class Word2VecStopsLoader(object):
    def __init__(self, clss):
        self.clss = clss

    def __iter__(self):
        notword = re.compile(r'[^\w -]')
        for i, sentence in enumerate(itertools.chain(*[session.query(cls).yield_per(100) for cls in self.clss])):
            filtered = [word for word in stemSentence(sentence.string) if len(notword.sub('', word)) > 0 and len(word) < 75]
            print i
            yield filtered

class TFIDFLoader(object):
    def __init__(self, clss):
        self.clss = clss

    def __iter__(self):
        tfidf = gensim.models.tfidfmodel.TfidfModel.load('/home/bbales2/scraping/webpage/db/{0}.tfidf.big'.format(prefix))
        for sentence in BOWLoader(self.clss):
            yield tfidf[sentence]

class LSILoader(object):
    def __init__(self, clss):
        self.clss = clss

    def __iter__(self):
        lsi = gensim.models.lsimodel.LsiModel.load('/home/bbales2/scraping/webpage/db/{0}.lsi.big'.format(prefix))
        for sentence in TFIDFLoader(self.clss):
            yield lsi[sentence]

class Doc2VecStopsLoader(object):
    def __init__(self, clss):
        self.clss = clss

    def __iter__(self):
        notword = re.compile(r'[^\w -]')
        for i, sentence in enumerate(itertools.chain(*[session.query(cls).order_by(cls.id).yield_per(100) for cls in self.clss])):
            filtered = [word for word in stemSentence(sentence.string) if len(notword.sub('', word)) > 0 and len(word) < 75]
            print i
            yield gensim.models.doc2vec.LabeledSentence(words = filtered, tags = [sentence.id])

#%%
notword = re.compile(r'[^\w -]')
for i, sentence in enumerate(session.query(mldb2.BodySentence).order_by(mldb2.BodySentence.id).yield_per(100)):
    stemmed = stemSentence(sentence.string)
    filtered = [word for word in stemmed if word not in stemmedStops and len(notword.sub('', word)) > 0 and len(word) < 75]
    print i
#%%
index2id = list(itertools.chain(*[[a[0] for a in session.query(cls).order_by(cls.id).with_entities(cls.id)] for cls in [mldb2.BodySentence, mldb2.AbstractSentence, mldb2.FigureSentence]]))
#%%
prefix = 'body_abstract_figure.xml'

with open('/home/bbales2/scraping/webpage/db/{0}.index2id.big'.format(prefix), 'w') as f:
    f.write(json.dumps(index2id))
#%%
#%%
#tmp = time.time()
#doc2vec = gensim.models.doc2vec.Doc2Vec()
#doc2vec.build_vocab(Doc2VecStopsLoader())
#doc2vec.train(Doc2VecStopsLoader())
#print 'Doc2Vec Training Time: ', time.time() - tmp
#doc2vec.save('/home/bbales2/scraping/webpage/db/doc2vec.big')
#%%
#%%
#corp = MyCorpus()

## Build per-sentence statistics
tmp = time.time()
word2vec = gensim.models.word2vec.Word2Vec(Word2VecStopsLoader([mldb2.BodySentence, mldb2.AbstractSentence, mldb2.FigureSentence]), workers = 4)
word2vec.save('/home/bbales2/scraping/webpage/db/{0}.word2vec.big.2'.format(prefix))
print 'Word2Vec Training Time: ', time.time() - tmp
#word2vec = gensim.models.word2vec.Word2Vec.load('/home/bbales2/scraping/webpage/db/{0}.word2vec.big'.format(prefix))
#%%
tmp = time.time()
dictionary = gensim.corpora.dictionary.Dictionary(LoaderStopwords([mldb2.BodySentence, mldb2.AbstractSentence, mldb2.FigureSentence]))
dictionary.filter_extremes(no_below = 5, no_above = 0.5)
dictionary.compactify()
print 'Dictionary time: ', time.time() - tmp
sys.stdout.flush()
dictionary.save('/home/bbales2/scraping/webpage/db/{0}.dictionary.big.2'.format(prefix))

import time
tmp = time.time()
tfidf = gensim.models.TfidfModel(BOWLoader([mldb2.BodySentence, mldb2.AbstractSentence, mldb2.FigureSentence]))
print 'Tfidf time: ', time.time() - tmp
sys.stdout.flush()
tfidf.save('/home/bbales2/scraping/webpage/db/{0}.tfidf.big.2'.format(prefix))

#dictionary = gensim.corpora.dictionary.Dictionary.load('/home/bbales2/scraping/webpage/db/dictionary.fig.big')

tmp = time.time()
numberTopics = 300
lsi = gensim.models.lsimodel.LsiModel(corpus = TFIDFLoader([mldb2.BodySentence, mldb2.AbstractSentence, mldb2.FigureSentence]), num_topics = numberTopics, id2word = dictionary)
print 'LSI Time: ', time.time() - tmp
lsi.save('/home/bbales2/scraping/webpage/db/{0}.lsi.big.2'.format(prefix))
sys.stdout.flush()

tmp = time.time()
index = gensim.similarities.docsim.Similarity('/home/bbales2/scraping/lsi_index/{0}.index.big.2'.format(prefix), LSILoader([mldb2.BodySentence, mldb2.AbstractSentence, mldb2.FigureSentence]), num_features = numberTopics)
print 'Index Time: ', time.time() - tmp
sys.stdout.flush()
index.save('/home/bbales2/scraping/webpage/db/{0}.index.big.2'.format(prefix))
#%%