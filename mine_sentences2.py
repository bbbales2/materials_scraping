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
import sqlalchemy
import sqlalchemy.ext.declarative
import lxml.etree
import itertools
import mldb2

from miningTools import stemSentence

f = open('/home/bbales2/scraping/rawData', 'r')#'/home/bbales2/scraping/electrochemistry/rawData'
rawData = json.loads(f.read())
f.close()

base = "/home/bbales2/scraping/data"#"/home/bbales2/scraping/electrochemistry/data"
import lxml.etree
import itertools

def dropns(root):
    for node in root.iter():
        if node.tag[0] == '{':
            node.tag = node.tag.rsplit('}', 1)[-1]

#%%
session, engine = mldb2.getSession('/home/bbales2/scraping/webpage/db/db.db')
#%%
worked = 0
for i, fname in enumerate(rawData):
    if i > 10:
        break

    xmlF = os.path.join(base, fname, "xml.xml")

    if os.path.exists(xmlF):
        print 'processing {0}'.format(xmlF)
        doc = ''
        with open(xmlF) as f:
            doc = f.read()

        etree = lxml.etree.fromstring(doc)

        dropns(etree)

        for e in itertools.chain(etree.findall( './/cross-ref' ),
                                 etree.findall( './/cross-refs' ),
                                etree.findall( './/formula' )):
        #                            etree.findall( './/sup' ),
        #                            etree.findall( './/hsp' ),
        #                            etree.findall( './/label' ),
        #                            etree.findall( './/inf' ),
        #                            #etree.findall( './/italic' )
            tail = e.tail
            e.clear()
            e.tail = tail
        #     elem.getparent().remove(elem)

        def get_text1(node):
            result = node.text or ""
            for child in node:
                if child.tail is not None:
                    result += child.tail
            return result

        def get_text3(node):
            return (node.text or "") + "".join(
                [etree.tostring(child) for child in node.iterchildren()])


        attachments = {}
        for attachment in etree.findall('.//attachment'):
             attachments[attachment.findtext('file-basename')] = attachment.findtext('attachment-eid')

        figures = []
        for figure in etree.findall('.//figure'):
            if len(figure.findall('.//figure')) == 0:
                figures.append( { 'caption' : ''.join(list(figure.itertext())[1:]),
                             'filename' : attachments[figure.find('link').get('locator')] } )

        abstract = ' '.join([''.join(para.itertext()) for para in etree.findall('.//abstract-sec')])

        doctext = []
        for para in etree.findall('.//para'):
            if len(para.findall('.//para')) == 0:
                doctext.append(''.join(para.itertext()))

        doctext = ' '.join(doctext)

        if len(doctext) == 0:
            continue

        worked += 1

        paperDb = session.query(mldb2.Paper).filter_by(name = fname).first()

        if paperDb:
            print 1/0

        paperdb = mldb2.Paper(name = fname, data = rawData[fname])
        session.add(paperdb)
        session.commit()

        abstractdb = mldb2.Abstract(paperId = paperdb.id)
        session.add(abstractdb)
        session.commit()

        tree = pattern.en.parsetree(abstract, lemmata = True)
        for j, sentence in enumerate(tree):
            session.add(mldb2.AbstractSentence(loc = j, string = sentence.string, abstractId = abstractdb.id))
        session.commit()

        tree = pattern.en.parsetree(doctext, lemmata = True)
        for j, sentence in enumerate(tree):
            session.add(mldb2.BodySentence(loc = j, string = sentence.string, paperId = paperdb.id))
        session.commit()

        for figure in figures:
            figdb = mldb2.Figure(filename = figure['filename'], paperId = paperdb.id)
            session.add(figdb)
            session.commit()

            tree = pattern.en.parsetree(figure['caption'], lemmata = True)
            for j, sentence in enumerate(tree):
                session.add(mldb2.FigureSentence(loc = j, string = sentence.string, figureId = figdb.id))
            session.commit()
        #print figures
        #print abstract
        #print doctext
    print '{0}/{1}'.format(i, len(rawData))