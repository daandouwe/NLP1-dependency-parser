import random
from sklearn.externals import joblib
import numpy as np


pos_tags = ['WRB', 'VBP', '-LRB-', '#', 'CC', 'IN', '.', 'WP', 'MD', 'RP', 'FW', 'CD', 'VBN', 'JJS', 'RBR', '``', 'PDT', 'WP$', ':', 'PRP', 'JJ', "''", 'NNPS', 'TO', 'JJR', 'RBS', 'VBD', 'VB', 'VBZ', 'NNP', 'DT', '$', 'UH', 'RB', 'EX', 'NNS', '-RRB-', 'POS', 'VBG', 'WDT', 'SYM', ',', 'LS', 'PRP$', 'NN']
labels = ['root', 'dep', 'aux', 'auxpass', 'cop', 'arg', 'agent', 'comp', 'acomp', 'ccomp', 'xcomp', 'obj', 'dobj', 'iobj', 'pobj', 'subj', 'nsubj', 'nsubjpass', 'csubj', 'csubjpass', 'cc', 'conj', 'expl', 'mod', 'amod', 'appos', 'advcl', 'det', 'predet', 'preconj', 'vmod', 'mwe', 'mark', 'advmod', 'neg', 'rcmod', 'quantmod', 'nn', 'npadvmod', 'tmod', 'num', 'number', 'prep', 'poss', 'possessive', 'prt', 'parataxis', 'goeswith', 'punct', 'ref', 'sdep', 'xsubj', 'partmod', 'abbrev', 'attr', 'complm', 'discourse', 'infmod', 'purpcl', 'rel', 'pcomp']

all = pos_tags + labels

tags_labels_to_vec = {}

for item in all:
    #tags_labels_to_vec[item] = [random.gauss(0, 10) for e in range(50)]
    #tags_labels_to_vec[item] = list(np.random.uniform(-1, 1, size=50))
    tags_labels_to_vec[item] = list(np.random.normal(loc=0.0, scale=1, size=50))


print(tags_labels_to_vec)