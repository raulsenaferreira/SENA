import numpy as np
from scipy.stats.stats import pearsonr
from scipy import stats
import matplotlib.pyplot as plt



def perform_corr_analysis():
	X=[]
	y=[]
	class_num = [i for i in range(0,10)]
	for c in class_num:
		h = np.load('results\\H\\hardness_{}.txt.npy'.format(c))
		n = np.load('results\\N\\nonconformity_{}.txt.npy'.format(c))
		arr_is_learnt = np.load('results\\I\\is_learned_{}.txt.npy'.format(c))
		conf = np.load('results\\C\\confidences_{}.txt.npy'.format(c))
		#lbl = [c]*len(n)
		#lbl = np.asarray(lbl)
		#checking if all arrays have the same length
		#print(len(h), len(n), len(arr_is_learnt), len(conf), len(lbl))
		
		#analyzing relashionship between hardness and nonconformity	
		print('results for class {}; len h: {}; len n: {}; pearson: {}; spearman: {}'.format(c, len(h), len(n), pearsonr(h,n), stats.spearmanr(h,n)))
		#analyzing relashionship between confidence and nonconformity	
		print('results for class {}; len conf: {}; len n: {}; pearson: {}; spearman: {}'.format(c, len(conf), len(n), pearsonr(conf,n), stats.spearmanr(conf,n)))
		#analyzing relashionship between hardness and confidence	
		print('results for class {}; len h: {}; len conf: {}; pearson: {}; spearman: {}'.format(c, len(h), len(conf), pearsonr(h,conf), stats.spearmanr(h,conf)))
		#analyzing relashionship between learned and confidence	
		print('results for class {}; len is_learnt: {}; len conf: {}; pearson: {}; spearman: {}'.format(c, len(arr_is_learnt), len(conf), pearsonr(arr_is_learnt,conf), stats.spearmanr(arr_is_learnt,conf)))
