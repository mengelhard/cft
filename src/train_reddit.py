import numpy as np
import pandas as pd
import tensorflow as tf
import itertools
import datetime
import os

from sklearn.metrics import roc_auc_score

from model_reddit import CFTModelReddit
from train_mimic import rae_over_samples, rae, ci


REDDIT_DIR = '/scratch/mme4/reddit'
#REDDIT_DIR = '/Users/mme/projects/cft/data/reddit_subset'


def main():

	train_fns, val_fns, test_fns = get_files(REDDIT_DIR)

	utc = datetime.datetime.utcnow().strftime('%s')

	n_outputs = 9

	results_fn = os.path.join(
		os.path.split(os.getcwd())[0],
		'results',
		'reddit_' + utc + '.csv')

	head = ['status', 'fpr', 'n_samples', 'gs_temperature',
			'hidden_layer_size', 'estimator', 'censoring_factor', 'n_iter',
			'final_train_nll', 'final_val_nll']
	head += ['mean_auc'] + [('auc%i' % i) for i in range (n_outputs)]
	head += ['mean_raem'] + [('raem%i' % i) for i in range(n_outputs)]
	head += ['mean_raea'] + [('raea%i' % i) for i in range(n_outputs)]
	head += ['mean_ci'] + [('ci%i' % i) for i in range(n_outputs)]

	with open(results_fn, 'w+') as results_file:
		print(', '.join(head), file=results_file)

	params = dict()

	for i in range(20):

		censoring_factor = [2., 3.][i % 2]

		#params['fpr'] = np.random.rand() + 1e-3
		params['fpr'] = .7
		#params['n_samples'] = int(np.random.rand() * 100 + 20)
		params['n_samples'] = 100
		#params['gs_temperature'] = np.random.rand() + 1e-2
		params['gs_temperature'] = .3
		#hidden_layer_size = int(np.random.rand() * 1000 + 100)
		hidden_layer_size = 750
		params['encoder_layer_sizes'] = (hidden_layer_size, )
		params['decoder_layer_sizes'] = (hidden_layer_size, )

		params['estimator'] = 'gs'

		# if i < 30:
		# 	params['estimator'] = 'gs'
		# else:
		# 	params['estimator'] = 'none'

		print('running with params:', params)

		#try:
		n_iter, final_train_nll, final_val_nll, aucs, raes_median, raes_all, cis = train_cft(
			params, censoring_factor, train_fns, val_fns, test_fns)
		status = 'complete'

		# except:
		# 	n_iter, final_train_nll, final_val_nll = [np.nan] * 3
		# 	aucs = [np.nan] * n_outputs
		# 	raes_median = [np.nan] * n_outputs
		# 	raes_all = [np.nan] * n_outputs
		# 	cis = [np.nan] * n_outputs
		# 	status = 'failed'

		results = [status, params['fpr'], params['n_samples'],
				   params['gs_temperature'],
				   params['encoder_layer_sizes'][0],
				   params['estimator'],
				   censoring_factor,
				   n_iter, final_train_nll,
				   final_val_nll]
		results += [np.nanmean(aucs)] + aucs
		results += [np.nanmean(raes_median)] + raes_median
		results += [np.nanmean(raes_all)] + raes_all
		results += [np.nanmean(cis)] + cis

		results = [str(r) for r in results]

		with open(results_fn, 'a') as results_file:
			print(', '.join(results), file=results_file)

		print('Run complete with status:', status)


def get_files(filedir):

	import glob
	filenames = glob.glob(filedir + '/*.pickle')
	filenames = np.random.RandomState(seed=0).permutation(filenames)

	print('There are %i files (approx %i total users)' % (
		len(filenames), 400 * len(filenames)))

	train_filenames = filenames[:int(.6 * len(filenames))]
	val_filenames = filenames[int(.6 * len(filenames)):int(.8 * len(filenames))]
	test_filenames = filenames[int(.8 * len(filenames)):]

	print(len(train_filenames), len(val_filenames), len(test_filenames))

	return train_filenames, val_filenames, test_filenames


def train_cft(model_params, censoring_factor, train_filenames, val_filenames, test_filenames):

	tf.reset_default_graph()

	cft_mdl = CFTModelReddit(
		embedding_layer_sizes=(300,),
		fpr_likelihood=True,
		prop_fpr=True,
		dropout_pct=.5,
		censoring_factor=censoring_factor,
		**model_params)

	with tf.Session() as sess:
		train_stats, val_stats = cft_mdl.train(
			sess, train_filenames, val_filenames,
			100, max_epochs_no_improve=3, learning_rate=3e-4,
			verbose=False)
		c_pred_cft, t_pred_cft, c_val, t_val, s_val = cft_mdl.predict_c_and_t(
			sess, test_filenames)

	train_stats = list(zip(*train_stats))
	val_stats = list(zip(*val_stats))

	n_iter = train_stats[0][-1]
	final_train_nll = train_stats[1][-1]
	final_val_nll = val_stats[1][-1]

	n_out = np.shape(c_val)[1]

	aucs = [roc_auc_score(c_val[:, i], c_pred_cft[:, i]) for i in range(n_out)]
	raes = [rae_over_samples(t_val[:, i], s_val[:, i], t_pred_cft[..., i]) for i in range(n_out)]
	cis = [ci(t_val[:, i], s_val[:, i], t_pred_cft[..., i]) for i in range(n_out)]

	raes_median, raes_all = list(zip(*raes))

	return n_iter, final_train_nll, final_val_nll, aucs, list(raes_median), list(raes_all), cis


if __name__ == '__main__':
	main()


