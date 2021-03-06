import numpy as np
import pandas as pd
import tensorflow as tf
import itertools
import datetime
import os

from sklearn.metrics import roc_auc_score

from model import mlp, lognormal_nlogpdf, lognormal_nlogsurvival
from model_reddit import load_batch
from train_reddit import get_files
from train_mimic import rae_over_samples, rae, ci

REDDIT_DIR = '/scratch/mme4/reddit'
#REDDIT_DIR = '/Users/mme/projects/cft/data/reddit_subset'
#REDDIT_DIR = '/home/rapiduser/dictionary_collection_2'


def main():

	train_fns, val_fns, test_fns = get_files(REDDIT_DIR)

	utc = datetime.datetime.utcnow().strftime('%s')

	n_outputs = 9

	results_fn = os.path.join(
		os.path.split(os.getcwd())[0],
		'results',
		'reddit_baselines_' + utc + '.csv')

	head = ['status', 'model_type', 'hidden_layer_size', 'censoring_factor',
			'n_iter', 'final_train_nll', 'final_val_nll']
	head += ['mean_auc'] + [('auc%i' % i) for i in range (n_outputs)]
	head += ['mean_raem'] + [('raem%i' % i) for i in range(n_outputs)]
	head += ['mean_raea'] + [('raea%i' % i) for i in range(n_outputs)]
	head += ['mean_ci'] + [('ci%i' % i) for i in range(n_outputs)]

	with open(results_fn, 'w+') as results_file:
		print(', '.join(head), file=results_file)

	for i in range(3 * 10 * 2):

		hidden_layer_sizes = (750, )
		model_type = ['survival', 'c_mlp', 's_mlp'][i % 3]
		censoring_factor = [2., 3.][i // 30]

		print('Running', model_type, 'with layers', hidden_layer_sizes)

		#try:
		n_iter, final_train_nll, final_val_nll, aucs, raes_median, raes_all, cis = train_baseline(
			model_type, censoring_factor, hidden_layer_sizes,
			train_fns, val_fns, test_fns)
		status = 'complete'

		# except:
		# 	n_iter, final_train_nll, final_val_nll = [np.nan] * 3
		# 	aucs = [np.nan] * n_outputs
		# 	raes_median = [np.nan] * n_outputs
		# 	raes_all = [np.nan] * n_outputs
		# 	cis = [np.nan] * n_outputs
		# 	status = 'failed'

		results = [status, model_type, hidden_layer_sizes[0],
				   censoring_factor, n_iter, final_train_nll,
				   final_val_nll]
		results += [np.nanmean(aucs)] + aucs
		results += [np.nanmean(raes_median)] + raes_median
		results += [np.nanmean(raes_all)] + raes_all
		results += [np.nanmean(cis)] + cis

		results = [str(r) for r in results]

		with open(results_fn, 'a') as results_file:
			print(', '.join(results), file=results_file)

		print('Run complete with status:', status)


def train_baseline(model_type, censoring_factor, hidden_layer_sizes, train_fns, val_fns, test_fns):

	tf.reset_default_graph()

	if model_type is 'survival':
		mdl = SurvivalModel(
			decoder_layer_sizes=hidden_layer_sizes, dropout_pct=.5,
			censoring_factor=censoring_factor)
	else:
		mdl = PosNegModel(
			encoder_layer_sizes=hidden_layer_sizes, dropout_pct=.5,
			censoring_factor=censoring_factor)

	with tf.Session() as sess:
		train_stats, val_stats = mdl.train(
			sess, train_fns, val_fns,
			100, train_type=model_type,
			max_epochs_no_improve=3, learning_rate=3e-4,
			verbose=False)
		predictions, c_val, t_val, s_val = mdl.predict(
			sess, test_fns)
		
	train_stats = list(zip(*train_stats))
	val_stats = list(zip(*val_stats))

	n_iter = train_stats[0][-1]
	final_train_nll = train_stats[1][-1]
	final_val_nll = val_stats[1][-1]

	n_out = np.shape(c_val)[1]

	if model_type is 'survival':
		c_prob_pred = t_to_prob(predictions)
		raes = [rae_over_samples(t_val[:, i], s_val[:, i], predictions[..., i]) for i in range(n_out)]
		cis = [ci(t_val[:, i], s_val[:, i], predictions[..., i]) for i in range(n_out)]
		raes_median, raes_all = list(zip(*raes))
	else:
		c_prob_pred = predictions
		raes_median = [np.nan] * n_out
		raes_all = [np.nan] * n_out
		cis = [np.nan] * n_out

	aucs = [roc_auc_score(c_val[:, i], c_prob_pred[:, i]) for i in range(n_out)]

	return n_iter, final_train_nll, final_val_nll, aucs, list(raes_median), list(raes_all), cis


def time_to_prob(x):
	sorted_pos = {v: pos for pos, v in enumerate(sorted(x))}
	return 1 - np.array([sorted_pos[v] / len(x) for v in x])


def t_to_prob(x):
	return np.stack([time_to_prob(arr) for arr in x.T]).T


class SurvivalModel:

	def __init__(self,
		embedding_layer_sizes=(),
		decoder_layer_sizes=(),
		dropout_pct=0.,
		censoring_factor=2.,
		activation_fn=tf.nn.relu):

		self.embedding_layer_sizes = embedding_layer_sizes
		self.decoder_layer_sizes = decoder_layer_sizes
		self.dropout_pct = dropout_pct
		self.censoring_factor = censoring_factor
		self.activation_fn = activation_fn


	def train(self, sess,
			  train_files,
			  val_files,
			  max_epochs,
			  train_type='survival',
			  max_epochs_no_improve=0,
			  learning_rate=1e-3,
			  batch_size=300, batch_eval_freq=1,
			  verbose=False):

		assert train_type is 'survival'

		self.n_outputs = 9
		self.opt = tf.train.AdamOptimizer(learning_rate)
		self._build_placeholders()
		self._build_x()
		self._build_model()

		sess.run(tf.global_variables_initializer())

		train_stats = []
		val_stats = []
		best_val_nloglik = np.inf
		n_epochs_no_improve = 0

		batches_per_epoch = len(train_files)

		for epoch_idx in range(max_epochs):

			for batch_idx, batch_file in enumerate(train_files):

				xvb, xfb, cb, tb, sb = load_batch(
					batch_file, censoring_factor=self.censoring_factor)

				lgnrm_nlogp_, lgnrm_nlogs_, _ = sess.run(
					[self.lgnrm_nlogp, self.lgnrm_nlogs, self.train_op],
					feed_dict={self.xv: xvb, self.xf: xfb, self.t: tb, self.s: sb, self.is_training: True})

				if np.isnan(np.mean(lgnrm_nlogp_)):
					print('Warning: lgnrm_nlogp is NaN')
				if np.isnan(np.mean(lgnrm_nlogs_)):
					print('Warning: lgnrm_nlogs is NaN')

				if batch_idx % batch_eval_freq == 0:
					idx = epoch_idx * batches_per_epoch + batch_idx
					train_stats.append(
						(idx, ) + self._get_train_stats(
							sess, xvb, xfb, tb, sb))

			idx = (epoch_idx + 1) * batches_per_epoch

			current_val_stats = []

			for val_batch_idx, batch_file in enumerate(val_files):

				xvb, xfb, cb, tb, sb = load_batch(
					batch_file, censoring_factor=self.censoring_factor)

				current_val_stats.append(
					self._get_train_stats(
						sess, xvb, xfb, tb, sb))

			print('current val stats are:', current_val_stats)

			val_stats.append((idx, ) + tuple(np.mean(current_val_stats, axis=0)))

			print('Completed Epoch %i' % epoch_idx)

			if verbose:
				self._summarize(
					np.mean(train_stats[-batches_per_epoch:], axis=0),
					val_stats[-1],
					batches_per_epoch)

			if val_stats[-1][1] < best_val_nloglik:
				best_val_nloglik = val_stats[-1][1]
				n_epochs_no_improve = 0
			else:
				n_epochs_no_improve += 1

			if n_epochs_no_improve > max_epochs_no_improve:
				break

		return train_stats, val_stats


	def _build_model(self):

		self.t_mu, self.t_logvar = self._decoder(self.x)

		nll = self._nloglik(self.t_mu, self.t_logvar)
		self.nll = tf.reduce_mean(nll)

		self.train_op = self.opt.minimize(self.nll)


	def _nloglik(self, t_mu, t_logvar):

		self.lgnrm_nlogp = lognormal_nlogpdf(self.t, self.t_mu, self.t_logvar)
		self.lgnrm_nlogs = lognormal_nlogsurvival(self.t, self.t_mu, self.t_logvar)

		nll = self.s * self.lgnrm_nlogp + (1 - self.s) * self.lgnrm_nlogs
		return nll


	def _build_placeholders(self):
		
		self.xv = tf.placeholder(
			shape=(None, 20, 512),
			dtype=tf.float32)

		self.xf = tf.placeholder(
			shape=(None, 2),
			dtype=tf.float32)

		self.t = tf.placeholder(
			shape=(None, self.n_outputs),
			dtype=tf.float32)

		self.s = tf.placeholder(
			shape=(None, self.n_outputs),
			dtype=tf.float32)

		self.is_training = tf.placeholder(
			shape=(),
			dtype=tf.bool)


	def _build_x(self):

		with tf.variable_scope('embeddings'):

			x_refined = mlp(
				self.xv, self.embedding_layer_sizes,
				dropout_pct=0.,
				activation_fn=tf.nn.tanh)

			x_max = tf.reduce_max(x_refined, axis=1)
			x_mean = tf.reduce_mean(x_refined, axis=1)

			self.x = tf.concat([x_max, x_mean, self.xf], axis=1)


	def _decoder(self, h, reuse=False):

		with tf.variable_scope('decoder', reuse=reuse):

			hidden_layer = mlp(
				h, self.decoder_layer_sizes,
				dropout_pct=self.dropout_pct,
				activation_fn=self.activation_fn,
				training=self.is_training,
				reuse=reuse)

			mu = tf.layers.dense(
				hidden_layer, self.n_outputs,
				activation=None,
				name='mu',
				reuse=reuse)

			logvar = tf.layers.dense(
				hidden_layer, self.n_outputs,
				activation=None,
				name='logvar',
				reuse=reuse)

		self.t_pred = tf.exp(mu)

		self.decoder_vars = tf.get_collection(
			tf.GraphKeys.GLOBAL_VARIABLES,
			scope='decoder')

		return mu, logvar


	def _get_train_stats(self, sess, xvs, xfs, ts, ss):

		nloglik, mean, logvar = sess.run(
			[self.nll,
			 self.t_mu,
			 self.t_logvar],
			feed_dict={self.xv: xvs, self.xf: xfs, self.t: ts, self.s:ss, self.is_training: False})

		return nloglik, np.mean(mean), np.mean(logvar)


	def _summarize(self, train_stats, val_stats, batches_per_epoch):

		print('nloglik (train) = %.2e' % train_stats[1])
		print('t_mu: %.2e' % train_stats[2], 't_logvar: %.2e' % train_stats[3])
		print('nloglik (val) = %.2e' % val_stats[1])
		print('t_mu: %.2e' % val_stats[2], 't_logvar: %.2e\n' % val_stats[3])


	def predict(self, sess, batch_files):

		t_pred = []
		c = []
		t = []
		s = []

		for idx, batch_file in enumerate(batch_files):

			xvb, xfb, cb, tb, sb = load_batch(
				batch_file, censoring_factor=self.censoring_factor)

			t_pred_ = sess.run(
				self.t_pred,
				feed_dict={self.xv: xvb, self.xf: xfb, self.is_training: False})

			t_pred.append(t_pred_)

			c.append(cb)
			t.append(tb)
			s.append(sb)

		t_pred = np.concatenate(t_pred, axis=0)
		c = np.concatenate(c, axis=0)
		t = np.concatenate(t, axis=0)
		s = np.concatenate(s, axis=0)

		return t_pred, c, t, s


class PosNegModel: ## amend this to use c vs s

	def __init__(self,
		embedding_layer_sizes=(),
		encoder_layer_sizes=(),
		dropout_pct=0.,
		censoring_factor=2.,
		activation_fn=tf.nn.relu):

		self.embedding_layer_sizes = embedding_layer_sizes
		self.encoder_layer_sizes = encoder_layer_sizes
		self.dropout_pct = dropout_pct
		self.censoring_factor = censoring_factor
		self.activation_fn = activation_fn


	def train(self, sess,
			  train_files,
			  val_files,
			  max_epochs,
			  train_type='s_mlp', max_epochs_no_improve=1,
			  learning_rate=1e-3,
			  batch_size=300, batch_eval_freq=10,
			  verbose=False):

		self.train_type = train_type
		self.n_outputs = 9
		self.opt = tf.train.AdamOptimizer(learning_rate)
		self._build_placeholders()
		self._build_x()
		self._build_model()

		sess.run(tf.global_variables_initializer())

		train_stats = []
		val_stats = []
		best_val_nloglik = np.inf
		n_epochs_no_improve = 0

		batches_per_epoch = len(train_files)

		for epoch_idx in range(max_epochs):

			for batch_idx, batch_file in enumerate(train_files):

				xvb, xfb, cb, tb, sb = load_batch(
					batch_file, censoring_factor=self.censoring_factor)

				if train_type is 's_mlp':

					xvb, xfb, _, tb, sb = load_batch(
						batch_file, censoring_factor=self.censoring_factor)

				elif train_type is 'c_mlp':

					xvb, xfb, sb, tb, _ = load_batch(
						batch_file, censoring_factor=self.censoring_factor)

				sess.run(
					self.train_op,
					feed_dict={self.xv: xvb, self.xf: xfb, self.s: sb, self.is_training: True})

				if batch_idx % batch_eval_freq == 0:
					idx = epoch_idx * batches_per_epoch + batch_idx
					train_stats.append((idx, self._get_train_stats(sess, xvb, xfb, sb)))

			idx = (epoch_idx + 1) * batches_per_epoch

			current_val_stats = []

			for val_batch_idx, batch_file in enumerate(val_files):

				if train_type is 's_mlp':

					xvb, xfb, _, tb, sb = load_batch(
						batch_file, censoring_factor=self.censoring_factor)

				elif train_type is 'c_mlp':

					xvb, xfb, sb, tb, _ = load_batch(
						batch_file, censoring_factor=self.censoring_factor)

				current_val_stats.append(self._get_train_stats(sess, xvb, xfb, sb))

			val_stats.append((idx, np.mean(current_val_stats)))

			print('Completed Epoch %i' % epoch_idx)

			if verbose:
				self._summarize(
					np.mean(train_stats[-batches_per_epoch:], axis=0),
					val_stats[-1],
					batches_per_epoch)

			if val_stats[-1][1] < best_val_nloglik:
				best_val_nloglik = val_stats[-1][1]
				n_epochs_no_improve = 0
			else:
				n_epochs_no_improve += 1

			if n_epochs_no_improve > max_epochs_no_improve:
				break

		return train_stats, val_stats


	def _build_model(self):

		self.s_logits, self.s_probs = self._encoder(self.x)

		nll = tf.nn.sigmoid_cross_entropy_with_logits(
			labels=self.s,
			logits=self.s_logits)

		self.nll = tf.reduce_mean(nll)

		self.train_op = self.opt.minimize(self.nll)


	def _build_placeholders(self):
		
		self.xv = tf.placeholder(
			shape=(None, 20, 512),
			dtype=tf.float32)

		self.xf = tf.placeholder(
			shape=(None, 2),
			dtype=tf.float32)

		self.s = tf.placeholder(
			shape=(None, self.n_outputs),
			dtype=tf.float32)

		self.is_training = tf.placeholder(
			shape=(),
			dtype=tf.bool)


	def _build_x(self):

		with tf.variable_scope('embeddings'):

			x_refined = mlp(
				self.xv, self.embedding_layer_sizes,
				dropout_pct=0.,
				activation_fn=tf.nn.tanh)

			x_max = tf.reduce_max(x_refined, axis=1)
			x_mean = tf.reduce_mean(x_refined, axis=1)

			self.x = tf.concat([x_max, x_mean, self.xf], axis=1)


	def _encoder(self, h):

		with tf.variable_scope('encoder'):

			hidden_layer = mlp(
				h, self.encoder_layer_sizes,
				dropout_pct=self.dropout_pct,
				training=self.is_training,
				activation_fn=self.activation_fn)

			logits = tf.layers.dense(
				hidden_layer, self.n_outputs,
				activation=None, name='logit_weights')

			probs = tf.nn.sigmoid(logits)

		self.encoder_vars = tf.get_collection(
			tf.GraphKeys.GLOBAL_VARIABLES,
			scope='encoder')

		return logits, probs


	def _get_train_stats(self, sess, xvs, xfs, ss):

		nloglik = sess.run(
			self.nll,
			feed_dict={self.xv: xvs, self.xf: xfs, self.s:ss, self.is_training: False})

		return nloglik


	def _summarize(self, train_stats, val_stats, batches_per_epoch):

		print('nloglik (train) = %.2e' % train_stats)
		print('nloglik (val) = %.2e' % val_stats)


	def predict(self, sess, batch_files):

		s_probs = []
		c = []
		t = []
		s = []

		for idx, batch_file in enumerate(batch_files):

			xvb, xfb, cb, tb, sb = load_batch(
				batch_file, censoring_factor=self.censoring_factor)

			s_probs_ = sess.run(
				self.s_probs,
				feed_dict={self.xv: xvb, self.xf: xfb, self.is_training: False})

			s_probs.append(s_probs_)

			c.append(cb)
			t.append(tb)
			s.append(sb)

		s_probs = np.concatenate(s_probs, axis=0)
		c = np.concatenate(c, axis=0)
		t = np.concatenate(t, axis=0)
		s = np.concatenate(s, axis=0)

		return s_probs, c, t, s


if __name__ == '__main__':
	main()
