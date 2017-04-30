import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from math import sqrt


def largest_indices(ary, n):
    '''Returns the n largest indices from a numpy array.'''
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

def plot_weights(trained_model_path, savepath, cmap='Greys', k=200):
	'''Plots reshaped weight vectors of given trained model
	as well as the sum of the absolute values of all the weight
	vectors.'''
	trained_classifier = joblib.load(trained_model_path)
	weights = trained_classifier.coefs_
	W = np.array(weights[0]).T
	matrix = np.abs(W[0])
	for i in range(1,200):
		matrix += np.abs(W[i])
	final = matrix.reshape((48,50)).T
	# plot the sum of the absolute values of all the weight vectors:
	plt.imshow(final, interpolation='nearest', cmap='Greys')
	plt.savefig(savepath.format('_')+'abssum')
	if cmap == 'RdBu':
		sgn = -1
	else:
		sgn = 1
	for i in range(k):
		v = W[i]
		v = sgn*np.array(v)
		final = v.reshape((48,50))
		plt.imshow(final.T, interpolation='nearest', cmap=cmap)
		plt.savefig(savepath.format(i))
		plt.clf()

def plot_best_entries_weights(trained_model_path, savepath, k, cmap='Greys', rang=None):
	'''Plots k best entries of reshaped weight vectors of given trained model.'''
	trained_classifier = joblib.load(trained_model_path)
	weights = trained_classifier.coefs_
	W = np.array(weights[0]).T
	matrix = np.abs(W[0])
	for i in range(1,200):
		matrix += np.abs(W[i])
	final = matrix.reshape((50,48))
	plt.imshow(final, interpolation='nearest', cmap=cmap)
	plt.savefig(savepath)
	for k in rang:
		ind = largest_indices(final, k)
		tuples = list(zip(list(ind[0]),list(ind[1])))
		print('Best {0} values are at: {1}'.format(k, tuples))
		matrix = np.zeros(final.shape)
		for t in tuples:
			value = final[t]
			matrix[t] = value
		if cmap == 'RdBu':
			matrix = -1*matrix
		plt.imshow(matrix, interpolation='nearest', cmap=cmap)
		plt.savefig(savepath.format(k))
		plt.clf()

plot_weights('models/trained/trained-9dec/fullcorpus_labeled_1epochs_trained_model.pkl',
			 savepath='plotted-weights/trained-9dec/epoch-1/weight-node{}',
			 cmap='RdBu',
			 k=200)




