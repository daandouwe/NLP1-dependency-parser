import random

pos_tags = ['WRB', 'VBP', '-LRB-', '#', 'CC', 'IN', '.', 'WP', 'MD', 'RP', 'FW', 'CD', 'VBN', 'JJS', 'RBR', '``', 'PDT', 'WP$', ':', 'PRP', 'JJ', "''", 'NNPS', 'TO', 'JJR', 'RBS', 'VBD', 'VB', 'VBZ', 'NNP', 'DT', '$', 'UH', 'RB', 'EX', 'NNS', '-RRB-', 'POS', 'VBG', 'WDT', 'SYM', ',', 'LS', 'PRP$', 'NN']
labels = ['root', 'dep', 'aux', 'auxpass', 'cop', 'arg', 'agent', 'comp', 'acomp', 'ccomp', 'xcomp', 'obj', 'dobj', 'iobj', 'pobj', 'subj', 'nsubj', 'nsubjpass', 'csubj', 'csubjpass', 'cc', 'conj', 'expl', 'mod', 'amod', 'appos', 'advcl', 'det', 'predet', 'preconj', 'vmod', 'mwe', 'mark', 'advmod', 'neg', 'rcmod', 'quantmod', 'nn', 'npadvmod', 'tmod', 'num', 'number', 'prep', 'poss', 'possessive', 'prt', 'parataxis', 'goeswith', 'punct', 'ref', 'sdep', 'xsubj', 'partmod', 'abbrev', 'attr', 'complm', 'discourse', 'infmod', 'purpcl', 'rel', 'pcomp']
#
# all = pos_tags + labels
#
# tags_labels_to_vec = {}
#
# for item in all:
#     tags_labels_to_vec[item] = [random.gauss(0, 2.5) for e in range(50)]
#
# print(tags_labels_to_vec)


tags_to_vec = {}
labels_to_vec = {}

count = 0
for tag in pos_tags:
    tags_to_vec[tag] = 50*[0]
    tags_to_vec[tag][count] = 1
    count += 1

#print(tags_to_vec)


twelve_least_frequent = ['cop', 'csubjpass', 'purpcl', 'abbrev', 'preconj', 'predet', 'csubj', 'rel', 'iobj', 'expl', 'mwe', 'parataxis']
labels_without_twelve_least_frequent = []

for label in labels:
    if not label in twelve_least_frequent:
        labels_without_twelve_least_frequent.append(label)

count = 0
for label in labels_without_twelve_least_frequent:
    labels_to_vec[label] = 50*[0]
    labels_to_vec[label][count] = 1
    count += 1
count = 1.0
for label in twelve_least_frequent:
    labels_to_vec[label] = 50*[0]
    labels_to_vec[label][49] = 1/count
    count += 1.0

tags_to_vec.update(labels_to_vec)
print(tags_to_vec)





