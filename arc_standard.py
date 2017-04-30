######################################

##### ARC STANDARD CONFIGURATION #####

######################################



from collections import deque
from glove_loader import *



class Stack:

    def __init__(self):

        self.contents = deque(['0'])



    def __str__(self):

        return str(self.contents)



    def add_to_stack(self, word):

        self.contents.append(word)



class Buffer:

    def __init__(self):

        self.contents = deque()



    def __str__(self):

        return str(self.contents)



    def load_to_buffer(self, sentence):

        if type(sentence) == list:

            self.contents.extend(sentence)

        elif type(sentence) == str:

            words = sentence.split()

            self.contents.extend(words)



    def shift_from_buffer(self):

        self.contents.popleft()



class Arcs:

    def __init__(self):

        self.labeled_arcs = []



    def __str__(self):

        return str(self.labeled_arcs)



    def unlabeled_arcs(self):

        return(list(map(lambda triple: (triple[0], triple[1]), self.labeled_arcs)))



    def add_to_labeled_arcs(self, triple):

        self.labeled_arcs.append(triple)



    def contains(self, head, dependent):

        # check if (head, dependent, label) in arcs for any label

        unlabeled_arcs = self.unlabeled_arcs()

        if (head, dependent) in unlabeled_arcs:

            return True

        else:

            return False



    def child_still_has_children(self, child):

        # check if word has no dependents of its own before being reduced

        unlabeled_arcs = list(self.unlabeled_arcs())

        (parents, children) = zip(*unlabeled_arcs)

        if not child in parents:

            return False

        else:

            return True



    def get_label(self, head, dependent):

        index_arc = self.unlabeled_arcs().index((head, dependent))

        label = self.labeled_arcs[index_arc][2]

        return(label)



class Configuration:

    def __init__(self, representation_mode, word_vec_dict, tags_labels_embedding, amputated_CM_features):

        self.stack = Stack()

        self.buffer = Buffer()

        self.arcs_right = Arcs()

        self.arcs_left = Arcs()

        self.tags_labels_embedding = tags_labels_embedding

        self.amputated_CM_features = amputated_CM_features

        self.word_vec_dict = word_vec_dict



        # mapping from original indices in sentence to (word, pos) tuples

        self.mapping = [('ROOT', 'NONE')]



        self.representation_mode = representation_mode



    def __str__(self):

        stack_string = 'Stack: ' + str(self.stack)

        buffer_string = 'Buffer: ' + str(self.buffer)

        arc_string = 'Arcs: ' + str(self.arcs_left.labeled_arcs + self.arcs_right.labeled_arcs)

        return(stack_string + '\n' + buffer_string + '\n' + arc_string)



    def repr(self):

        """

        extracts information from configuration according to the specified representation mode



        :return: list of chosen features

        """

        if self.representation_mode == 'default':

            stack_string = 'Stack: ' + str(self.stack)

            buffer_string = 'Buffer: ' + str(self.buffer)

            arc_string = 'Arcs: ' + str(self.arcs_left.labeled_arcs + self.arcs_right.labeled_arcs)

            return(stack_string + '\n' + buffer_string + '\n' + arc_string)



        if self.representation_mode == 'CM':

            # Chen&Manning style representation



            top3_stack = []

            top3_stack_pos = []

            for i in [-1, -2, -3]:

                try:

                    # top 3 words on stack

                    top3_stack += [self.mapping[int(self.stack.contents[i])][0]]



                    # POS tags of top 3 words on stack

                    top3_stack_pos += [self.mapping[int(self.stack.contents[i])][1]]

                except:

                    top3_stack += ['_']

                    top3_stack_pos += ['_']



            top3_buffer = []

            top3_buffer_pos = []

            for i in [0, 1, 2]:

                try:

                    # top 3 words on buffer

                    top3_buffer += [self.mapping[int(self.buffer.contents[i])][0]]



                    # POS tags of top 3 words on buffer

                    top3_buffer_pos += [self.mapping[int(self.buffer.contents[i])][1]]

                except:

                    top3_buffer += ['_']

                    top3_buffer_pos += ['_']



            # first and second leftmost/rightmost children of the top two words on the stack

            leftmost = []

            leftmost_pos = []

            rightmost = []

            rightmost_pos = []



            # leftmost of leftmost/rightmost of rightmost children of the top two words on the stack

            leftleftmost = []

            leftleftmost_pos = []

            rightrightmost = []

            rightrightmost_pos = []



            # arc labels

            leftmost_labels = []

            leftleftmost_labels = []

            rightmost_labels = []

            rightrightmost_labels = []



            for i in [-1, -2]:

                iter_left = -1

                count_leftmost = 0

                count_leftleftmost = 0



                iter_right = -1

                count_rightmost = 0

                count_rightrightmost = 0



                if i == -2 and len(self.stack.contents) < 2:

                    # catch case where stack only has one element

                    leftmost += (2 - count_leftmost) * ['_']

                    leftmost_pos += (2 - count_leftmost) * ['_']

                    leftmost_labels += (2 - count_leftmost) * ['_']



                    leftleftmost += (1 - count_leftleftmost) * ['_']

                    leftleftmost_pos += (1 - count_leftleftmost) * ['_']

                    leftleftmost_labels += (1 - count_leftleftmost) * ['_']

                    rightmost += (2 - count_rightmost) * ['_']

                    rightmost_pos += (2 - count_rightmost) * ['_']

                    rightmost_labels += (2 - count_rightmost) * ['_']



                    rightrightmost += (1 - count_rightrightmost) * ['_']

                    rightrightmost_pos += (1 - count_rightrightmost) * ['_']

                    rightrightmost_labels += (1 - count_rightrightmost) * ['_']



                else:

                    while  count_leftmost < 2 and (-iter_left) <= len(self.arcs_left.unlabeled_arcs()):

                        if self.arcs_left.unlabeled_arcs()[iter_left][0] == self.stack.contents[i]:

                            leftmost += [self.mapping[int(self.arcs_left.unlabeled_arcs()[iter_left][1])][0]]

                            leftmost_pos += [self.mapping[int(self.arcs_left.unlabeled_arcs()[iter_left][1])][1]]

                            leftmost_labels += [self.arcs_left.get_label(self.arcs_left.unlabeled_arcs()[iter_left][0], self.arcs_left.unlabeled_arcs()[iter_left][1])]

                            count_leftmost += 1



                            if count_leftmost == 1:

                                iter_leftleft = -1

                                while len(leftleftmost) < 1 and (-iter_leftleft) <= len(self.arcs_left.unlabeled_arcs()):

                                    if self.arcs_left.unlabeled_arcs()[iter_leftleft][0] == self.arcs_left.unlabeled_arcs()[iter_left][1]:

                                        leftleftmost += [self.mapping[int(self.arcs_left.unlabeled_arcs()[iter_leftleft][1])][0]]

                                        leftleftmost_pos += [self.mapping[int(self.arcs_left.unlabeled_arcs()[iter_leftleft][1])][1]]

                                        leftleftmost_labels += [self.arcs_left.get_label(self.arcs_left.unlabeled_arcs()[iter_leftleft][0], self.arcs_left.unlabeled_arcs()[iter_leftleft][1])]

                                        count_leftleftmost += 1

                                    iter_leftleft -= 1

                        iter_left -= 1



                    leftmost += (2 - count_leftmost) * ['_']

                    leftmost_pos += (2 - count_leftmost) * ['_']

                    leftmost_labels += (2 - count_leftmost) * ['_']



                    leftleftmost += (1 - count_leftleftmost) * ['_']

                    leftleftmost_pos += (1 - count_leftleftmost) * ['_']

                    leftleftmost_labels += (1 - count_leftleftmost) * ['_']





                    while len(rightmost) < 2 and (-iter_right) <= len(self.arcs_right.unlabeled_arcs()):

                        if self.arcs_right.unlabeled_arcs()[iter_right][0] == self.stack.contents[i]:

                            rightmost += [self.mapping[int(self.arcs_right.unlabeled_arcs()[iter_right][1])][0]]

                            rightmost_pos += [self.mapping[int(self.arcs_right.unlabeled_arcs()[iter_right][1])][1]]

                            rightmost_labels += [self.arcs_right.get_label(self.arcs_right.unlabeled_arcs()[iter_right][0], self.arcs_right.unlabeled_arcs()[iter_right][1])]

                            count_rightmost += 1



                            if len(rightmost) == 1:

                                iter_rightright = -1

                                while len(rightrightmost) < 1 and (-iter_rightright) <= len(self.arcs_right.unlabeled_arcs()):

                                    if self.arcs_right.unlabeled_arcs()[iter_rightright][0] == self.arcs_right.unlabeled_arcs()[iter_right][1]:

                                        rightrightmost += [self.mapping[int(self.arcs_right.unlabeled_arcs()[iter_rightright][1])][0]]

                                        rightrightmost_pos += [self.mapping[int(self.arcs_right.unlabeled_arcs()[iter_rightright][1])][1]]

                                        rightrightmost_labels += [self.arcs_right.get_label(self.arcs_right.unlabeled_arcs()[iter_rightright][0], self.arcs_right.unlabeled_arcs()[iter_rightright][1])]

                                        count_rightrightmost += 1

                                    iter_rightright -= 1

                        iter_right -= 1



                    rightmost += (2 - count_rightmost) * ['_']

                    rightmost_pos += (2 - count_rightmost) * ['_']

                    rightmost_labels += (2 - count_rightmost) * ['_']



                    rightrightmost += (1 - count_rightrightmost) * ['_']

                    rightrightmost_pos += (1 - count_rightrightmost) * ['_']

                    rightrightmost_labels += (1 - count_rightrightmost) * ['_']





            words = top3_stack + top3_buffer + leftmost + rightmost + leftleftmost + rightrightmost

            pos_tags = top3_stack_pos + top3_buffer_pos + leftmost_pos + rightmost_pos + leftleftmost_pos + rightrightmost_pos

            arc_labels = leftmost_labels + rightmost_labels + leftleftmost_labels + rightrightmost_labels



            features = words + pos_tags + arc_labels



            if not self.word_vec_dict == None:

                # convert features to vector representation

                vector_words = [self.word_vec_dict[string.lower()] if string.lower() in self.word_vec_dict else 50*[0] for string in words]

                vector_pos_tags = [self.tags_labels_embedding[string] for string in pos_tags]

                vector_arc_labels = [self.tags_labels_embedding[string] for string in arc_labels]

                vector_features = vector_words + vector_pos_tags + vector_arc_labels



                # delete features at specified indices

                counter = 0

                for amputated_feature_index in self.amputated_CM_features:

                    del vector_features[amputated_feature_index-counter]

                    counter += 1

                return(vector_features)



            return(features)



    def load_sentence(self, sentence):

        self.buffer.load_to_buffer(sentence)



    def shift(self):

        self.stack.add_to_stack(self.buffer.contents[0])

        self.buffer.shift_from_buffer()



    def left_reduce(self, label):

        self.arcs_left.add_to_labeled_arcs((self.stack.contents[-1], self.stack.contents[-2], label))

        del self.stack.contents[-2]



    def right_reduce(self, label):

        self.arcs_right.add_to_labeled_arcs((self.stack.contents[-2], self.stack.contents[-1], label))

        del self.stack.contents[-1]



    def all_labeled_arcs(self):

        return(self.arcs_right.labeled_arcs + self.arcs_left.labeled_arcs)

