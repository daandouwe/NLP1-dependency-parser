from sklearn.externals import joblib

#######################################
## SAVED AS PKL FILES AT /EMBEDDINGS ##
#######################################

# One-hot:
tags_labels_dict_onehot = joblib.load('embeddings/one-hot/tags_labels_dict_onehot')

# Trained
tags_labels_dict_trained9dec = joblib.load('embeddings/trained/tags_labels_dict_trained9dec')
tags_labels_dict_trained11dec = joblib.load('embeddings/trained/tags_labels_dict_trained11dec')

