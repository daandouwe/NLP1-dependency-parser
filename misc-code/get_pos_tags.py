# file = open('data/train-stanford-raw.conll', 'r')
#
# #list_sentences = []
# pos_tags = []
#
# idx = 0
# for line in file:
#     if line != '\n':
#         line_split = line.split('\t')
#         pos_tags.append(line_split[4])
#
# pos_tags = list(set(pos_tags))
# print(pos_tags)
# print(len(pos_tags))
#
# ['WRB', 'VBP', '-LRB-', '#', 'CC', 'IN', '.', 'WP', 'MD', 'RP', 'FW', 'CD', 'VBN', 'JJS', 'RBR', '``', 'PDT', 'WP$', ':', 'PRP', 'JJ', "''", 'NNPS', 'TO', 'JJR', 'RBS', 'VBD', 'VB', 'VBZ', 'NNP', 'DT', '$', 'UH', 'RB', 'EX', 'NNS', '-RRB-', 'POS', 'VBG', 'WDT', 'SYM', ',', 'LS', 'PRP$', 'NN']

file = open('labels', 'r')
labels = []
# numerical_transition = 0
#
for line in file:
     label = line.split('\n')[0]
     labels.append(label)
print(labels)
#
#     numerical_transition += 1
#
#     labels_dict[label] = numerical_transition
#     number_to_label[numerical_transition] = label
#
# #ordered = sorted(labels_dict)
# print(labels_dict)
# print(number_to_label)