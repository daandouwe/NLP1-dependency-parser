from arc_standard import *

from glove_loader import *



def analyse_conll(data_file):

    """

    analyses training data in CoNLL format



    :param data_file: file with training data in CoNLL format

    :return: list of triples, one for each sentence:

    (indices, word_pos, arcs_given),

    where:

    indices: indices of words in sentence

    word_pos: list of tuples containing (word, pos-tag) of index

    arcs_given: list dependency relations as labeled in training data

    """



    file = open(data_file, 'r')



    data_triples = []

    list_sentences = [[]]



    idx = 0

    for line in file:

        if line == '\n':

            idx += 1

            list_sentences.append([])

        else:

            list_sentences[idx].append(line)



    for sentence in list_sentences:

        arcs_given = Arcs()

        indices = []

        word_pos = []

        for line in sentence:

            line_split = line.split('\t')

            indices.append(line_split[0])

            word_pos.append((line_split[1], line_split[4]))

            arcs_given.labeled_arcs.append((line_split[6], line_split[0], line_split[7]))

        data_triples.append((indices, word_pos, arcs_given))

    return(data_triples)



def get_numerical_transition(direction, reduction):

    """

    converts transition into corresponding numerical representation



    :param direction: left or right

    :param reduction: label of arc

    :return: numerical representation of the transition/new labeled arc

    """



    labels_to_numerical_representation = {'expl': 23, 'rcmod': 36, 'appos': 26, 'agent': 7, 'mod': 24, 'nsubjpass': 18, 'possessive': 45, 'arg': 6, 'advmod': 34, 'number': 42, 'neg': 35, 'partmod': 53, 'acomp': 9, 'prep': 43, 'auxpass': 4, 'amod': 25, 'csubjpass': 20, 'goeswith': 48, 'ref': 50, 'quantmod': 37, 'aux': 3, 'punct': 49, 'parataxis': 47, 'conj': 22, 'mwe': 32, 'purpcl': 59, 'comp': 8, 'obj': 12, 'preconj': 30, 'poss': 44, 'nsubj': 17, 'nn': 38, 'det': 28, 'sdep': 51, 'dobj': 13, 'discourse': 57, 'mark': 33, 'prt': 46, 'csubj': 19, 'npadvmod': 39, 'dep': 2, 'pobj': 15, 'infmod': 58, 'subj': 16, 'attr': 55, 'iobj': 14, 'cc': 21, 'ccomp': 10, 'predet': 29, 'xcomp': 11, 'abbrev': 54, 'tmod': 40, 'root': 1, 'cop': 5, 'pcomp': 61, 'xsubj': 52, 'rel': 60, 'advcl': 27, 'vmod': 31, 'num': 41, 'complm': 56}

    number = labels_to_numerical_representation[reduction]

    if direction == 'left':

        return number*-1

    if direction == 'right':

        return number



def get_label_transition(number):

    """

    converts numerical representation into corresponding transition



    :param number: numerical representation of classification class (transition label)

    :return: corresponding label as string

    """



    numerical_representation_to_label = {1: 'root', 2: 'dep', 3: 'aux', 4: 'auxpass', 5: 'cop', 6: 'arg', 7: 'agent', 8: 'comp', 9: 'acomp', 10: 'ccomp', 11: 'xcomp', 12: 'obj', 13: 'dobj', 14: 'iobj', 15: 'pobj', 16: 'subj', 17: 'nsubj', 18: 'nsubjpass', 19: 'csubj', 20: 'csubjpass', 21: 'cc', 22: 'conj', 23: 'expl', 24: 'mod', 25: 'amod', 26: 'appos', 27: 'advcl', 28: 'det', 29: 'predet', 30: 'preconj', 31: 'vmod', 32: 'mwe', 33: 'mark', 34: 'advmod', 35: 'neg', 36: 'rcmod', 37: 'quantmod', 38: 'nn', 39: 'npadvmod', 40: 'tmod', 41: 'num', 42: 'number', 43: 'prep', 44: 'poss', 45: 'possessive', 46: 'prt', 47: 'parataxis', 48: 'goeswith', 49: 'punct', 50: 'ref', 51: 'sdep', 52: 'xsubj', 53: 'partmod', 54: 'abbrev', 55: 'attr', 56: 'complm', 57: 'discourse', 58: 'infmod', 59: 'purpcl', 60: 'rel', 61: 'pcomp'}

    label = numerical_representation_to_label[number]

    return(label)



def infer_trans_sentence(indices, word_pos, arcs_given, representation, word_vec_dict, tags_labels_embedding, amputated_CM_features):

    """

    computes the sequence of configurations and corresponding transitions that an ideal

    parser would perform in order to generate the dependency structure of a sentence as

    specified in the training data



    :param indices: indices of words in sentence

    :param word_pos: list of tuples containing (word, pos-tag) of index

    :param arcs_given: list dependency relations as labeled in training data

    :param representation: chosen mode of representation

    :param word_vec_dict: dictionary mapping words to 50-dimensional vectors

    :param tags_labels_embedding: dictionary mapping POS tags and arc labels to 50-dimensional vectors

    :param amputated_CM_features: list of indices of Chen&Manning style features to be taken out of consideration

    :return: tuple (configurations, transitions),

    where

    configurations: list of configurations in chosen mode of representation

    transitions: list of transitions in numerical representation

    """



    # initialise configuration with chosen representation mode

    config = Configuration(representation, word_vec_dict, tags_labels_embedding, amputated_CM_features)

    config.mapping.extend(word_pos)

    config.load_sentence(indices)

    configurations = [config.repr()]



    # always start with one shift

    config.shift()

    transitions = [0]



    # (almost) always followed by a second shift

    if len(config.buffer.contents) > 0:

        configurations.append(config.repr())

        config.shift()

        transitions.append(0)



    # decide which transition to take based on the arc structure given in the training data

    num_transitions = 1

    while len(config.stack.contents) > 1:

        if num_transitions == 250:

            break

        num_transitions += 1

        configurations.append(config.repr())



        stack_first = config.stack.contents[-1]

        stack_second = config.stack.contents[-2]



        if not stack_second == '0':

            if arcs_given.contains(stack_first, stack_second):

                

                # check if word has no children before reducing it

                if not arcs_given.child_still_has_children(stack_second):

                    label = arcs_given.get_label(stack_first, stack_second)

                    transitions.append(get_numerical_transition('left', label))

                    config.left_reduce(label)

                    arcs_given.labeled_arcs.remove((stack_first, stack_second, label))



                elif len(config.buffer.contents) > 0:

                    transitions.append(0)

                    config.shift()



            elif arcs_given.contains(stack_second, stack_first):

                

                if not arcs_given.child_still_has_children(stack_first):

                    label = arcs_given.get_label(stack_second, stack_first)

                    transitions.append(get_numerical_transition('right', label))

                    config.right_reduce(label)

                    arcs_given.labeled_arcs.remove((stack_second, stack_first, label))



                elif len(config.buffer.contents) > 0:

                    transitions.append(0)

                    config.shift()

                    

            elif len(config.buffer.contents) > 0:

                transitions.append(0)

                config.shift()

                

        elif len(config.buffer.contents) > 0:

            transitions.append(0)

            config.shift()



        else:

            # perform final right-reduction

            label = arcs_given.get_label(stack_second, stack_first)

            transitions.append(get_numerical_transition('right', label))

            config.right_reduce(label)



    return(configurations, transitions)

