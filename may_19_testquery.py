#%%
import gensim
import json
from miningTools import stemSentence
import numpy
import os
import stemming.porter2
#import mldb
import sqlalchemy.orm
import itertools
import mldb2
import subprocess

import re
#%%


session, engine = mldb2.getSession()

#%%

for i, paper in  enumerate(session.query(mldb2.Paper).all()):
    if re.sub('[-\(\)]', '', paper.data['pii']) == 'S1359646212000243':
        1/0
    print i

#%%
pId = 7335

paper = session.query(mldb2.Paper).get(pId)

os.chdir('/home/bbales2/models/syntaxnet')

processed = {}

for sentence in paper.sentences:
#if True:
#    sentence = session.query(mldb2.BodySentence).get(1118658)#1118700
    #string = re.sub(sentence.string
    #print sentence.string
    string = re.sub('(?P<name>[0-9\.]+)', '\g<name> ', sentence.string)
    string = re.sub(u'\u2013', ' to ', string)
    #print string

    #1/0

    h = subprocess.Popen('echo "{0}" | syntaxnet/demo.sh'.format(string), shell = True, stderr = subprocess.PIPE, stdout = subprocess.PIPE)

    stdout, stderr = h.communicate()

    words = []
    for line in stdout.split('\n'):
        line = line.strip()

        if len(line) > 0:
            idx, word, _, upos, _, _, head, dep, _, _ = line.split('\t')
            words.append({ 'word' : word,
                           'pos' : upos,
                           'head' : int(head) - 1,
                           'dep' : dep })


    processed[sentence.id] = words
#%%
for docId in processed:
    for wdict in processed[docId]:
        if re.search('\xc2\xb0C', wdict['word']):
            1/0
#%%
#sentence = session.query(mldb2.BodySentence).get(1118700)
sentence = session.query(mldb2.BodySentence).get(1118658)

print [sentence.string]
#%%
print "\n".join([str(p) for p in processed[1118658]])
#%%
for sId, sentence in processed.items():
    for w in sentence:
        #if 'precipitate' in w['word']:
        #    print sId
        if w['head'] >= 0:
            if w['dep'] == 'num':
                #if re.search('m', sentence[w['head']]['word']):
                #    print 'hi'
                #if re.search('C', sentence[w['head']]['word']):
                if re.search('Pa', sentence[w['head']]['word']):
                    #print w
                    print w['word'], sentence[w['head']]['word']
                if re.search('\xce\xbcm', sentence[w['head']]['word']) or re.search('mm', sentence[w['head']]['word']) or re.search('nm', sentence[w['head']]['word']):
                    #print w
                    print w['word'], sentence[w['head']]['word']
           # else:
            #    if re.search('m', w['word']):
            #        print w['word'], sId

#%%

#session.query(mldb2.Abstract).get(1)
#session.query(mldb2.FigureSentence).get(1).string
#%%
for cls in [mldb2.BodySentence, mldb2.AbstractSentence, mldb2.FigureSentence]:
    print session.query(cls).count()
#index2id = list(itertools.chain(*[[a[0] for a in session.query(cls).order_by(cls.id).with_entities(cls.id)] ]))
#%%

with open('/home/bbales2/scraping/webpage/db/index2id.fig.big') as f:
    index2id = json.load(f)
#list(session.query(mldb.Sentence).options(sqlalchemy.orm.load_only("id")).order_by(mldb.Sentence.id).with_entities(mldb.Sentence.id))

index = gensim.similarities.docsim.MatrixSimilarity.load('/home/bbales2/scraping/webpage/db/index.fig.big')
dictionary = gensim.corpora.dictionary.Dictionary.load('/home/bbales2/scraping/webpage/db/dictionary.fig.big')
tfidf = gensim.models.tfidfmodel.TfidfModel.load('/home/bbales2/scraping/webpage/db/tfidf.fig.big')
lsi = gensim.models.lsimodel.LsiModel.load('/home/bbales2/scraping/webpage/db/lsi.fig.big')
word2vec = gensim.models.word2vec.Word2Vec.load('/home/bbales2/scraping/webpage/db/word2vec.big')
#%%

positiveWords = [stemming.porter2.stem(word) for word in 'Al Co'.split()]
radius = 1

#for state in states:
result = index[lsi[tfidf[[dictionary.doc2bow(positiveWords)]]]].flatten()

rankings = { 'lsi' : [],
             'doc2vec' : [] }



#cs = numpy.concatenate((numpy.zeros(radius + 1), numpy.cumsum(result), numpy.zeros(radius)))

#integrated = numpy.zeros(len(result))
#for i in range(len(result)):
#    integrated[i] = cs[i + 2 * radius + 1] - cs[i]

#for i in numpy.argsort(-integrated)[0:5]:
#    rankings['lsi'].append((index2id[i][0], integrated[i]))
for i in numpy.argsort(-result)[0:5]:
    rankings['lsi'].append((index2id[i][0], result[i]))
#result[i]
        #positiveVec = doc2vec.infer_vector(positiveWords)
        #negativeVec = doc2vec.infer_vector(negativeWords)

        #rankings['doc2vec'] = []
        #for idx, score in doc2vec.docvecs.most_similar(positive = [positiveVec], negative = [negativeVec], topn = 10):
        #    rankings['doc2vec'].append((idx, 0.0))

sentenceIdxs = rankings['lsi']#set([idx for idx, score in rankings['lsi']])# | set([idx for idx, score in rankings['doc2vec']])

paperIds = set()

sentences = []
for idx, weight in sorted(sentenceIdxs, key = lambda x : x[1], reverse = True):
    id_ = index2id[idx][0] - 1
    if id_ >= 1328461:
        print "error"

    sentence = session.query(mldb2.BodySentence).get(id_)
    sentences.append({
        'id' : index2id[idx][0],
        'string' : sentence.string,
        'weight' : weight,
        'paperId' : sentence.paperId
    })

    paperIds.add(sentence.paperId)
