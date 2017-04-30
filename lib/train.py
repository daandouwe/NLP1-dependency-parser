import numpy as np
import process_data as pd
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from arc_standard import *
from glove_loader import *
from embeddings import tags_labels_dict_onehot, tags_labels_dict_trained9dec, tags_labels_dict_trained11dec

class ModelTrainer:

    def __init__(self, 
                 glove_file,
                 data_train, 
                 representation,
                 save_location, 
                 classes=[i for i in range(-61, 62)],
                 pretrained_model_path=None,
                 save_every_k_words=False,
                 tags_labels_embedding=None,
                 amputated_CM_features = []):
        self.glove = glove_file
        self.data = data_train
        self.repr = representation
        self.save_location = save_location
        self.word_vec_dict = load_glove_model(self.glove)
        self.classes = classes
        self.pretrained_model_path = pretrained_model_path
        self.save_every_k_words = save_every_k_words
        self.tags_labels_embedding = tags_labels_embedding
        self.amputated_CM_features = amputated_CM_features

        # CM configuration: alpha (called 'lamnbda' in CM) = 10^-8, hidden layer size = 200
        # from scikit-learn documentation: "MLPClassifier supports only the Cross-Entropy loss function".
        # Good, this is the same as CM.
        if self.pretrained_model_path == None:
            self.classifier = MLPClassifier(solver='sgd', alpha=1e-8,
                                            hidden_layer_sizes=(200,),
                                            activation='tanh')
        else:
            print('Using pretrained model at {}'.format(self.pretrained_model_path))
            self.classifier = joblib.load(self.pretrained_model_path)

        print('Testing save path')
        joblib.dump('testje', self.save_location)
        print('...')
        print('Test succesful. Please manually remove test-file {0}'.format(self.save_location))
        if self.amputated_CM_features != []:
            print('Amputated features: {0}'.format(self.amputated_CM_features))
        else:
            print('Amputated features: []')

    def train(self, epochs=1, start_epoch=1):
        '''
        Get data for all sentences in two big lists: one input (configs), one targets (trans).
        Inputs are also merged to one big list per input, so no more separate lists for
        vectorised representations of different words.
        '''
        print('Training on data at {0}'.format(self.data))
        print('Training with solver `sgd`')
        epoch = start_epoch
        while epoch <= epochs:
            print('Training epoch {0}/{1}'.format(epoch, epochs))
            sentence_count = 0
            for (indices, word_pos, arcs_given) in pd.analyse_conll(self.data):
                sentence_count += 1
                (configurations, transitions_batch) = pd.infer_trans_sentence(indices, word_pos, arcs_given, self.repr, self.word_vec_dict, self.tags_labels_embedding, self.amputated_CM_features)
                configurations_batch = []

                for configuration in configurations:
                    conf_one_list = []
                    for word in configuration:
                        for scalar in word:
                            conf_one_list += [scalar]
                    configurations_batch.append(conf_one_list)

                if sentence_count%100 == 0:
                    print('Training on sentence {0}'.format(sentence_count))

                # Train the classifier on the batches.
                self.classifier.partial_fit(configurations_batch, transitions_batch, self.classes)

                if type(self.save_every_k_words) == int:
                    if sentence_count%self.save_every_k_words == 0:
                        print('Partially trained model pickled and saved at {0} sentence count is {1}'.format(self.save_location, sentence_count))
                        joblib.dump(self.classifier, self.save_location)

            joblib.dump(self.classifier, self.save_location.format(epoch))
            print('Epoch {0} trained model pickled and saved'.format(epoch))
            epoch += 1
        print('Training finished')


####################################
######## ONE-HOT embeddings ########
####################################

# m = ModelTrainer('/Users/Daan/Documents/Logic/NLP1/depparser-labeled-final/glove/glove.6B.50d.txt',
#                  'data/train-stanford-raw.conll',
#                  'CM',
#                  'models/fullcorpus_labeled/random_POS_embedding/one-hot/fullcorpus_labeled_randomPOS_{0}epochs_trained_model.pkl',
#                  tags_labels_embedding=tags_labels_dict_onehot,
#                  pretrained_model_path='models/fullcorpus_labeled/random_POS_embedding/one-hot/fullcorpus_labeled_randomPOS_8epochs_trained_model.pkl')

# m.train(epochs = 12, start_epoch=9)


################################
####### REMOVED FEATURES #######
################################


# amp_feat_best = [0,1,18,19]
amp_feat_worst = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 44, 45, 46, 47]

m = ModelTrainer('/Users/Daan/Documents/Logic/NLP1/depparser-labeled-final/glove/glove.6B.50d.txt',
                 'data/train-stanford-raw.conll',
                 'CM',
                 'models/one-hot/worst-features-amputated/fullcorpus_labeled_{0}epochs_trained_model.pkl',
                 tags_labels_embedding=tags_labels_dict_onehot,
                 amputated_CM_features=amp_feat_worst)

m.train(epochs=12)





