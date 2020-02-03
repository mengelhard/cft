import numpy as np
import pandas as pd
import tensorflow as tf
import itertools
import datetime
import os

from sklearn.metrics import roc_auc_score

from model_mimic import CFTModelMimic


def main():

	batch_indices = list(range(143))

	train_indices = batch_indices[:86]
	val_indices = batch_indices[86:114]
	test_indices = batch_indices[114:]

	utc = datetime.datetime.utcnow().strftime('%s')

	results_fn = os.path.join(
		os.path.split(os.getcwd())[0],
		'results',
		'mimic_' + utc + '.csv')

	head = ['status', 'fpr', 'n_samples', 'gs_temperature',
			'hidden_layer_size', 'estimator', 'n_iter',
			'final_train_nll', 'final_val_nll',
			'mean_auc', 'auc0', 'auc1', 'auc2', 'auc3',
			'auc4', 'auc5', 'auc6', 'auc7', 'auc8' ,'auc9']

	with open(results_fn, 'w+') as results_file:
		print(', '.join(head), file=results_file)

	params = dict()

	for i in range(25):

		params['fpr'] = np.random.rand() + 1e-3
		params['n_samples'] = int(np.random.rand() * 100 + 20)
		params['gs_temperature'] = np.random.rand() + 1e-2
		hidden_layer_size = int(np.random.rand() * 1000 + 100)
		params['encoder_layer_sizes'] = (hidden_layer_size, )
		params['decoder_layer_sizes'] = (hidden_layer_size, )

		if i < 20:
			params['estimator'] = 'gs'
		else:
			params['estimator'] = 'none'

		print('running with params:', params)

		try:
			n_iter, final_train_nll, final_val_nll, aucs = train_cft(
				params, train_indices, val_indices, test_indices)
			mean_auc = np.mean(aucs)
			status = 'complete'

		except:
			n_iter, final_train_nll, final_val_nll, mean_auc = [np.nan] * 4
			aucs = [np.nan] * 10
			status = 'failed'

		results = [status, params['fpr'], params['n_samples'],
				   params['gs_temperature'],
				   params['encoder_layer_sizes'][0],
				   params['estimator'],
				   n_iter, final_train_nll,
				   final_val_nll, mean_auc] + aucs

		results = [str(r) for r in results]

		with open(results_fn, 'a') as results_file:
			print(', '.join(results), file=results_file)

		print('Run complete with status:', status)


def train_cft(model_params, train_indices, val_indices, test_indices):

	tf.reset_default_graph()

	cft_mdl = CFTModelMimic(
		fpr_likelihood=True,
		prop_fpr=True,
		dropout_pct=.5,
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

	aucs = [roc_auc_score(c_val[:, i], c_pred_cft[:, i]) for i in range(np.shape(c_val)[1])]

	return n_iter, final_train_nll, final_val_nll, aucs


if __name__ == '__main__':
	main()
