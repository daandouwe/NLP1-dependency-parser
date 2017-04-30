import numpy as np
import process_data as pd
import os
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from arc_standard import *
from glove_loader import *
from embeddings import tags_labels_dict_onehot, tags_labels_dict_trained9dec, tags_labels_dict_trained11dec

class ModelTest:

    def __init__(self, 
                 trained_model_path,
                 word_vect_dict, 
                 representation, 
                 tags_labels_embedding,
                 amputated_CM_features = [],
                 classes=[i for i in range(-61, 62)]):
        self.trained_classifier = joblib.load(trained_model_path)
        self.word_vect_dict = word_vect_dict
        self.repr = representation
        self.classes = classes
        self.amputated_CM_features = amputated_CM_features
        self.tags_labels_embedding = tags_labels_embedding

    def predict_final_sentence_configurations_from_file(self, conll_in):
        '''
        Uses the trained_classifier to return the predicted transitions.
        Format: list of list of elements in {-1,0,1}. 
        This format can be turned into a conll file by pd.file_trans_to_conll.
        This is done in self.predict_conll below. 
        '''

        data_triples = pd.analyse_conll(conll_in)
        data_pairs = list(map(lambda triple: (triple[0], triple[1]), data_triples))
        all_final_configurations = []

        sentence_count = 1

        for (indices, word_pos) in data_pairs:
            sentence_count += 1
            if sentence_count%100 == 0:
                    print('Predicting sentence {0}'.format(sentence_count))

            # get initial configurations per sentence
            conf = Configuration('CM', self.word_vect_dict, self.tags_labels_embedding, self.amputated_CM_features)
            conf.load_sentence(indices)
            conf.mapping.extend(word_pos)

            predicted_transitions = []

            shift_count = 0
            reduction_count = 0

            while len(predicted_transitions) < (2 * len(indices)):

                conf_array = np.array(conf.repr()).reshape(1, -1)
                pred = self.trained_classifier.predict(conf_array)[0]

                if pred == 0:
                    # Get second best action if best action is not possible. Same as Chen and Manning.
                    if len(conf.buffer.contents) == 0 or shift_count == len(indices):
                        second_best = sorted(self.trained_classifier.predict_proba(conf_array).tolist()[0], reverse = True)[1]
                        index_second_best = self.trained_classifier.predict_proba(conf_array).tolist()[0].index(second_best)
                        second_best_class = self.trained_classifier.classes_[index_second_best]
                        if second_best_class > 0:
                            label = pd.get_label_transition(second_best_class)
                            conf.right_reduce(label)
                            reduction_count += 1
                            real_prediction = second_best_class

                        elif second_best_class < 0:
                            label = pd.get_label_transition(abs(second_best_class))
                            conf.left_reduce(label)
                            reduction_count += 1
                            real_prediction = second_best_class

                    else:
                        conf.shift()
                        shift_count += 1
                        real_prediction = 0
                elif pred > 0:
                    if len(conf.stack.contents) < 2 or reduction_count == len(indices):
                        # now standardly does shift if cannot right reduce, but perhaps left reduce is sometimes better?
                        # consider using predict_proba again to find best non-right-reduce transition

                        second_best = sorted(self.trained_classifier.predict_proba(conf_array).tolist()[0], reverse = True)[1]
                        index_second_best = self.trained_classifier.predict_proba(conf_array).tolist()[0].index(second_best)
                        second_best_class = self.trained_classifier.classes_[index_second_best]
                        if second_best_class == 0:
                            conf.shift()
                            shift_count += 1
                            real_prediction = 0

                        elif second_best_class < 0:
                            label = pd.get_label_transition(abs(second_best_class))
                            conf.left_reduce(label)
                            reduction_count += 1
                            real_prediction = second_best_class

                    else:
                        label = pd.get_label_transition(pred)
                        conf.right_reduce(label)
                        reduction_count += 1
                        real_prediction = pred
                elif pred < 0:
                    if (len(conf.stack.contents) < 2) or (conf.stack.contents[-2] == '0' and len(conf.buffer.contents) != 0) or (reduction_count == len(indices)):
                        second_best = sorted(self.trained_classifier.predict_proba(conf_array).tolist()[0], reverse = True)[1]
                        index_second_best = self.trained_classifier.predict_proba(conf_array).tolist()[0].index(second_best)
                        second_best_class = self.trained_classifier.classes_[index_second_best]
                        if second_best_class == 0:
                            conf.shift()
                            shift_count += 1
                            real_prediction = 0

                        elif second_best_class > 0:
                            label = pd.get_label_transition(second_best_class)
                            conf.right_reduce(label)
                            reduction_count += 1
                            real_prediction = second_best_class
                    else:
                        label = pd.get_label_transition(abs(pred))
                        conf.left_reduce(label)
                        reduction_count += 1
                        real_prediction = pred

                predicted_transitions += [real_prediction]
            all_final_configurations.append(conf)

        return(all_final_configurations)

    def final_configurations_to_conll(self, final_configurations, output_file):
        output_conll = open(output_file, 'w')

        sentence_count = 0
        for final_config in final_configurations:
            if not sentence_count == 0:
                output_conll.write('\n')
            sentence_count += 1

            all_labeled_arcs = final_config.arcs_left.labeled_arcs + final_config.arcs_right.labeled_arcs
            number_words = len(all_labeled_arcs)
            indices = list(range(number_words + 1))[1:]

            for word_index in indices:
                for arc in all_labeled_arcs:
                    if arc[1] == str(word_index):
                        output_conll.write(str(word_index) + '\t' + 'WORD' + '\t' + '_' + '\t' + 'TAG' + '\t' + 'TAG'+ '\t' + '_' + '\t'+ str(arc[0]) + '\t' + str(arc[2]) + '\t' + '_' + '\t' + '_' + '\n')

    def predict_conll(self, conll_in, conll_out):
        '''
        This does not 'return' anything, instead, it saves the predicted
        conll as a text file at the path given at conll_out.
        '''
        final_configurations = self.predict_final_sentence_configurations_from_file(conll_in)
        self.final_configurations_to_conll(final_configurations, conll_out)

    def eval(self, conll_in, conll_out, result_path):
        '''result_path should be 'results/filename_of_choice'.
        The .txt extension is added automatically.
        Returns results in filename_of_choice.txt.'''
        self.predict_conll(conll_in, conll_out)
        print('Prediction finished, predicted conll-file saved at {}'.format(conll_out))
        os.system('perl eval.pl -g {0} -s {1} > {2}.txt'.format(conll_in, conll_out, result_path))
        print('Evaluation finished, results saved at {}'.format(result_path))
        return(None)


###########################################
####### Looped testing of the model #######
###########################################

#tags_labels_dict = tags_labels_dict_trained9dec

def eval_loop(n_min, n_max):

    '''Performs all the prediction and evaluation (including saving)
    of the test model using a loop. NOTE: all path setttings need 
    to be changed manually inside the function-body.'''

    glove_path = '/Users/Daan/Documents/Logic/NLP1/depparser-labeled-final/glove/glove.6B.50d.txt'
    word_vect_dict = load_glove_model(glove_path)
    for n in range(n_min, n_max+1):
        print('Evaluating the epoch {} model'.format(n))
        ### SETTING TEST-MODEL PATHS ###
        #model_path = 'models/one-hot/all-features/fullcorpus_labeled_{0}epochs_trained_model.pkl'.format(n)
        model_path = 'models/trained/trained-9dec/fullcorpus_labeled_{0}epochs_trained_model.pkl'.format(n)

        ### CREATING TEST OBJECT ###
        test_model = ModelTest(model_path, word_vect_dict, representation='CM', 
                                                           tags_labels_embedding=tags_labels_dict, 
                                                           amputated_CM_features=[])

        ### SETTING TEST PATHS ###
        # The path of the conll file we would like to test on:
        conll_in = 'data/test-stanford-raw.conll'
        # The path where we want to save the predicted conll:
        conll_out = 'output-conll/test-set/trained/trained-9dec/fullcorpus_labeled_onehot_{}epochs_test-stanford-raw'.format(n)
        # The path where we want to save the evaluation txt-file:
        result_path = 'results/test-set/trained/trained-9dec/result_fullcorpus_labeled_onehot_{}epochs_test-stanford-raw'.format(n)

        ### EVALUATING THE MODEL ON TEST PATHS ###
        test_model.eval(conll_in, conll_out, result_path)

#eval_loop(n_min=12, n_max=12)

tags_labels_dict = tags_labels_dict_onehot
amp_feat_worst = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 44, 45, 46, 47]
amp_feat_best = [0,1,18,19]


def eval_loop2(n_min, n_max):

    '''Performs all the prediction and evaluation (including saving)
    of the test model using a loop. NOTE: all path setttings need 
    to be changed manually inside the function-body.'''

    glove_path = '/Users/Daan/Documents/Logic/NLP1/depparser-labeled-final/glove/glove.6B.50d.txt'
    word_vect_dict = load_glove_model(glove_path)
    for n in range(n_min, n_max+1):
        print('Evaluating the epoch {} model'.format(n))
        ### SETTING TEST-MODEL PATHS ###
        #model_path = 'models/one-hot/all-features/fullcorpus_labeled_{0}epochs_trained_model.pkl'.format(n)
        model_path = 'models/one-hot/worst-features-amputated/fullcorpus_labeled_{}epochs_trained_model.pkl'.format(n)

        ### CREATING TEST OBJECT ###
        test_model = ModelTest(model_path, word_vect_dict, representation='CM', 
                                                           tags_labels_embedding=tags_labels_dict, 
                                                           amputated_CM_features=amp_feat_worst)

        ### SETTING TEST PATHS ###
        # The path of the conll file we would like to test on:
        conll_in = 'data/test-stanford-raw.conll'
        # The path where we want to save the predicted conll:
        conll_out = 'output-conll/test-set/one-hot/best-features-amputated/fullcorpus_labeled_onehot_{}epochs_test-stanford-raw'.format(n)
        # The path where we want to save the evaluation txt-file:
        result_path = 'results/test-set/one-hot/best-features-amputated/result_fullcorpus_labeled_onehot_{}epochs_test-stanford-raw'.format(n)

        ### EVALUATING THE MODEL ON TEST PATHS ###
        test_model.eval(conll_in, conll_out, result_path)

eval_loop2(n_min=12, n_max=12)

def eval_loop3(n_min, n_max):

    '''Performs all the prediction and evaluation (including saving)
    of the test model using a loop. NOTE: all path setttings need 
    to be changed manually inside the function-body.'''

    glove_path = '/Users/Daan/Documents/Logic/NLP1/depparser-labeled-final/glove/glove.6B.50d.txt'
    word_vect_dict = load_glove_model(glove_path)
    for n in range(n_min, n_max+1):
        print('Evaluating the epoch {} model'.format(n))
        ### SETTING TEST-MODEL PATHS ###
        #model_path = 'models/one-hot/all-features/fullcorpus_labeled_{0}epochs_trained_model.pkl'.format(n)
        model_path = 'models/one-hot/all-features/fullcorpus_labeled_{}epochs_trained_model.pkl'.format(n)

        ### CREATING TEST OBJECT ###
        test_model = ModelTest(model_path, word_vect_dict, representation='CM', 
                                                           tags_labels_embedding=tags_labels_dict, 
                                                           amputated_CM_features=[])

        ### SETTING TEST PATHS ###
        # The path of the conll file we would like to test on:
        conll_in = 'data/test-stanford-raw.conll'
        # The path where we want to save the predicted conll:
        conll_out = 'output-conll/test-set/one-hot/all-features/fullcorpus_labeled_onehot_{}epochs_test-stanford-raw'.format(n)
        # The path where we want to save the evaluation txt-file:
        result_path = 'results/test-set/one-hot/all-features/result_fullcorpus_labeled_onehot_{}epochs_test-stanford-raw'.format(n)

        ### EVALUATING THE MODEL ON TEST PATHS ###
        test_model.eval(conll_in, conll_out, result_path)

#eval_loop3(n_min=15, n_max=15)

#############################################
####### 'Manual' testing of the model #######
#############################################

# ### LOADING THE GLOVE FILE ###
# glove_path = '/Users/Daan/Documents/Logic/NLP1/depparser-labeled-final/glove/glove.6B.50d.txt'
# word_vect_dict = load_glove_model(glove_path)

# ### SETTING TEST-MODEL PATHS ###
# model_path = 'models/smallcorpus/smallcorpus_1epochs_trained_model.pkl'
# #model_path = 'models/fullcorpus_2epoch_trained_model.pkl'
# representation = 'CM'

# ### CREATING TEST OBJECT ###
# test_model = ModelTest(model_path, word_vect_dict, representation)

# ### SETTING TEST PATHS ###
# # The path of the conll test we would like to test on:
# conll_in = 'data/disjunct_testset'
# # The path where we want to save the predicted conll:
# conll_out = 'output_conll/smallcorpus/smallcorpus_1epoch_disjunct_testset'
# # The path where we want to save the evaluation txt-file:
# result_path = 'results/smallcorpus/result_smallcorpus_1epoch_disjunct_testset'

# ### EVALUATING THE MODEL ON TEST PATHS ###
# test_model.eval(conll_in, conll_out, result_path)
