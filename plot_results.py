import matplotlib
import matplotlib.pyplot as plt
from numpy import add

matplotlib.style.use('ggplot')

def plot_results(results_path, savepath, n):
	las_epochs = []
	uas_epochs = []
	for i in range(1,n+1):
		txt = open(results_path.format(i), 'r')
		las_line = txt.readline()
		uas_line = txt.readline()
		las = las_line[-8:-3]
		uas = uas_line[-8:-3]
		las_epochs.append(las)
		uas_epochs.append(uas)
	epochs = range(1,n+1)
	ax1 = plt.plot(epochs, uas_epochs, label='UAS')
	ax2 = plt.plot(epochs, las_epochs, label='LAS')
	handles = ax1+ax2
	plt.legend(handles=handles, loc=4)
	plt.ylabel('% accuracy')
	plt.xlabel('Epochs')
	plt.savefig(savepath)


def plot_both_results(results_path1, results_path2, savepath, n, m):
	las_epochs = []
	uas_epochs = []
	label_epochs = []
	for i in range(1,n+1):
		txt = open(results_path1.format(i), 'r')
		las_line = txt.readline()
		uas_line = txt.readline()
		las = las_line[-8:-3]
		uas = uas_line[-8:-3]
		las_epochs.append(las)
		uas_epochs.append(uas)
	epochs = range(1,n+1)
	ax1 = plt.plot(epochs, uas_epochs, '--', label='UAS one-hot', color=plt.rcParams['axes.color_cycle'][0])
	ax3 = plt.plot(epochs, las_epochs, '--', label='LAS one-hot', color=plt.rcParams['axes.color_cycle'][1])
	las_epochs = []
	uas_epochs = []
	for i in range(1,m+1):
		txt = open(results_path2.format(i), 'r')
		las_line = txt.readline()
		uas_line = txt.readline()
		las = las_line[-8:-3]
		uas = uas_line[-8:-3]
		las_epochs.append(las)
		uas_epochs.append(uas)
	epochs = range(1,m+1)
	ax2 = plt.plot(epochs, uas_epochs, label='UAS trained', color=plt.rcParams['axes.color_cycle'][0])
	ax4 = plt.plot(epochs, las_epochs, label='LAS trained', color=plt.rcParams['axes.color_cycle'][1])
	handles = ax1+ax2+ax3+ax4
	plt.legend(handles=handles, loc=4)
	plt.ylabel('% accuracy')
	plt.xlabel('Epochs')
	plt.savefig(savepath, dpi=300)
	
def plot_amputation_histogram(results_path1, results_path2, results_path3, savepath):
	ylas = []
	yuas = []
	for results_path in [results_path1, results_path2, results_path3]:
		txt = open(results_path, 'r')
		las_line = txt.readline()
		uas_line = txt.readline()
		las = las_line[-8:-3]
		uas = uas_line[-8:-3]
		ylas.append(float(las))
		yuas.append(float(uas))
	x = range(3)
	width = [0.2, 0.2, 0.2]
	fig = plt.figure()
	ax = fig.add_subplot(111)
	uasbar = ax.bar(x, yuas, width=0.2, align='center', color=plt.rcParams['axes.color_cycle'][0], label='UAS')
	lasbar = ax.bar(add(list(x), width), ylas, width=0.2, align='center', color=plt.rcParams['axes.color_cycle'][1], label='LAS')
	labels = ['All features', 'Features\n[6-17, 24-35, 44-47]\nremoved', 'Features\n[0,1,18,19]\nremoved']
	ax.legend((uasbar[0], lasbar[0]), ('UAS', 'LAS'))
	plt.xticks(x, labels)
	plt.savefig(savepath, dpi=300)


# results_path1 = 'results/test-set/one-hot/all-features/result_fullcorpus_labeled_onehot_{}epochs_test-stanford-raw.txt'
# savepath = 'plots/test-set/results-onehot-test'
# plot_results(results_path1, savepath, 12)

results_path1 = 'results/test-set/one-hot/all-features/result_fullcorpus_labeled_onehot_{}epochs_test-stanford-raw.txt'
results_path2 = 'results/test-set/trained/trained-9dec/fullcorpus_labeled_one-hot_emb_{}epochs_testset.txt'
savepath = 'plots/test-set/results-onehot-trained-testset'

n = 12
m = 12
plot_both_results(results_path1, results_path2, savepath, n, m)

# results_path1 = 'results/test-set/one-hot/all-features/result_fullcorpus_labeled_onehot_12epochs_test-stanford-raw.txt'
# results_path2 = 'results/test-set/one-hot/worst-features-amputated/result_fullcorpus_labeled_onehot_12epochs_test-stanford-raw.txt'
# results_path3 = 'results/test-set/one-hot/best-features-amputated/result_fullcorpus_labeled_onehot_12epochs_test-stanford-raw.txt'
# savepath = 'plots/amp-feat-histogram'

# plot_amputation_histogram(results_path1, results_path2, results_path3, savepath)

