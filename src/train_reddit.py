import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
import datetime
import os

from sklearn.metrics import roc_auc_score

from model_reddit import CFTModelReddit
from baselines import SurvivalModel, PosNegModel


def main():

	train_fns, val_fns, test_fns = get_files()

	utc = datetime.datetime.utcnow().strftime('%s')

	results_fn = os.path.join(
		os.path.split(os.getcwd())[0],
		'results',
		'reddit_gs_' + utc + '.csv')

	head = ['status', 'fpr', 'n_samples', 'gs_temperature',
			'n_iter', 'final_train_nll', 'final_val_nll',
			'mean_auc', 'auc0', 'auc1', 'auc2', 'auc3',
			'auc4', 'auc5', 'auc6', 'auc7', 'auc8']

	with open(results_fn, 'w+') as results_file:
		print(', '.join(head), file=results_file)

	param_options = {
		'fpr': [1., .5, .1, .01],
		'n_samples': [30, 100],
		'gs_temperature': [.1, .3, 1.]
	}

	for params_list in itertools.product(
		param_options['fpr'],
		param_options['n_samples'],
		param_options['gs_temperature']):

		params = {
			'fpr': params_list[0],
			'n_samples': params_list[1],
			'gs_temperature': params_list[2]
		}

		print('running with params:', params)

		for i in range(5):

			try:
				n_iter, final_train_nll, final_val_nll, aucs = train_cft(
					params, train_fns, val_fns, test_fns)
				mean_auc = np.mean(aucs)
				status = 'complete'

			except:
				n_iter, final_train_nll, final_val_nll, mean_auc = [np.nan] * 4
				aucs = [np.nan] * 9
				status = 'failed'

			results = [status, params['fpr'], params['n_samples'],
					   params['gs_temperature'],
					   n_iter, final_train_nll,
					   final_val_nll, mean_auc] + aucs

			results = [str(r) for r in results]

			with open(results_fn, 'a') as results_file:
				print(', '.join(results), file=results_file)


def get_files():

	import glob
	filenames = glob.glob('/scratch/mme4/reddit/*.pickle')
	#filenames = glob.glob('../data/reddit_subset/*.pickle')
	filenames = np.random.RandomState(seed=0).permutation(filenames)

	print('There are %i files (approx %i total users)' % (
		len(filenames), 400 * len(filenames)))

	train_filenames = filenames[:int(.6 * len(filenames))]
	val_filenames = filenames[int(.6 * len(filenames)):int(.8 * len(filenames))]
	test_filenames = filenames[int(.8 * len(filenames)):]

	print(len(train_filenames), len(val_filenames), len(test_filenames))

	return train_filenames, val_filenames, test_filenames


def train_cft(model_params, train_filenames, val_filenames, test_filenames):

	tf.reset_default_graph()

	cft_mdl = CFTModelReddit(
		embedding_layer_sizes=(300,),
		encoder_layer_sizes=(100,),
		decoder_layer_sizes=(100,),
		estimator='gs',
		fpr_likelihood=True,
		prop_fpr=True,
		dropout_pct=.5,
		**model_params)

	with tf.Session() as sess:
		train_stats, val_stats = cft_mdl.train(
			sess, train_filenames, val_filenames,
			100, max_epochs_no_improve=5, learning_rate=3e-4,
			verbose=False)
		c_pred_cft, t_pred_cft, c_val, t_val, s_val = cft_mdl.predict_c_and_t(
			sess, val_filenames)
		
	train_stats = list(zip(*train_stats))
	val_stats = list(zip(*val_stats))

	n_iter = train_stats[0][-1]
	final_train_nll = train_stats[1][-1]
	final_val_nll = val_stats[1][-1]

	aucs = [roc_auc_score(c_val[:, i], c_pred_cft[:, i]) for i in range(9)]

	return n_iter, final_train_nll, final_val_nll, aucs


if __name__ == '__main__':
	main()


