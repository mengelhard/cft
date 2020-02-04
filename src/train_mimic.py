import numpy as np
import pandas as pd
import tensorflow as tf
import itertools
import datetime
import os

from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index

from model_mimic import CFTModelMimic


def main():

	batch_indices = list(range(143))

	#train_indices = batch_indices[:86]
	#val_indices = batch_indices[86:114]
	#test_indices = batch_indices[114:]

	train_indices = batch_indices[:2]
	val_indices = batch_indices[2:4]
	test_indices = batch_indices[4:6]

	n_outputs = 10

	utc = datetime.datetime.utcnow().strftime('%s')

	results_fn = os.path.join(
		os.path.split(os.getcwd())[0],
		'results',
		'mimic_' + utc + '.csv')

	head = ['status', 'fpr', 'n_samples', 'gs_temperature',
			'hidden_layer_size', 'estimator', 'n_iter',
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

		# if i < 20:
		# 	params['estimator'] = 'gs'
		# else:
		# 	params['estimator'] = 'none'

		print('running with params:', params)

		try:
			n_iter, final_train_nll, final_val_nll, aucs, raes_median, raes_all, cis = train_cft(
				params, censoring_factor, train_indices, val_indices, test_indices)
			status = 'complete'

		except:
			n_iter, final_train_nll, final_val_nll = [np.nan] * 3
			aucs = [np.nan] * n_outputs
			raes_median = [np.nan] * n_outputs
			raes_all = [np.nan] * n_outputs
			cis = [np.nan] * n_outputs
			status = 'failed'

		results = [status, params['fpr'], params['n_samples'],
				   params['gs_temperature'],
				   params['encoder_layer_sizes'][0],
				   params['estimator'],
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


def train_cft(model_params, censoring_factor, train_indices, val_indices, test_indices):

	tf.reset_default_graph()

	cft_mdl = CFTModelMimic(
		fpr_likelihood=True,
		prop_fpr=True,
		dropout_pct=.5,
		censoring_factor=censoring_factor,
		**model_params)

	with tf.Session() as sess:
		train_stats, val_stats = cft_mdl.train(
			sess, train_indices, val_indices,
			100, max_epochs_no_improve=3, learning_rate=3e-4,
			verbose=False)
		c_pred_cft, t_pred_cft, c_val, t_val, s_val = cft_mdl.predict_c_and_t(
			sess, val_indices)
		
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


def rae_over_samples(t_true, s, t_pred):

	if t_pred.ndim > 1:
		raes_median = rae(t_true, s, np.median(t_pred, axis=1))
		raes_all = rae(t_true[:, np.newaxis], s, t_pred)
		return raes_median, raes_all
	else:
		raes = rae(t_true, s, t_pred)
		return raes, raes


def rae(t_true, s, t_pred):
	errors = (t_true - t_pred) / t_true.max()
	errors_uncensored = np.abs(errors[s == 1].flatten())
	errors_censored = np.maximum(errors[s == 0].flatten(), 0)
	#print('max t is', t_true.max())
	#print('avg error (not censored) is', np.mean(errors_uncensored))
	#print('avg error (censored) is', np.mean(errors_censored))
	return np.mean(np.concatenate([errors_uncensored, errors_censored]))


def ci(t_true, s, t_pred):
	if t_pred.ndim > 1:
		try:
			cidx = concordance_index(t_true, np.median(t_pred, axis=1), s)
		except:
			cidx = np.nan
	else:
		try:
			cidx = concordance_index(t_true, t_pred, s)
		except:
			cidx = np.nan
	return cidx


if __name__ == '__main__':
	main()
