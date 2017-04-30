import subprocess
import os

conll_path1 = 'data/tinydata'
conll_path2 = 'data/tinydata'
filename = 'test4'

#os.system('perl eval.pl -g data/tinydata.conll -s data/tinydatapredict.conll > results.txt')
os.system('perl eval.pl -g {0} -s {1} > {2}.txt'.format(conll_path1, conll_path2, filename))
#subprocess.call('perl eval.pl -g data/tinydata.conll -s data/tinydatapredict.conll > results/results.txt')