import numpy as np
import tensorflow as tf
import pandas as pd
import itertools
import datetime
import os

from sklearn.metrics import roc_auc_score

from model import mlp, lognormal_nlogpdf, lognormal_nlogsurvival, generate_dataset, partition
from train_mimic import rae_over_samples, rae, ci


def main():

	N = 40000

	x_all, c_all, t_all, s_all = generate_dataset(N)

	assert not np.any(np.isnan(np.concatenate(
		[x_all.flatten(), c_all.flatten(), t_all.flatten(), s_all.flatten()])))

	x_train, x_val, x_test = partition(x_all, [.6, .8])
	c_train, c_val, c_test = partition(c_all, [.6, .8])
	t_train, t_val, t_test = partition(t_all, [.6, .8])
	s_train, s_val, s_test = partition(s_all, [.6, .8])

	print('Train data have shape', np.shape(x_train), np.shape(t_train), np.shape(s_train))
	print('Val data have shape', np.shape(x_val), np.shape(t_val), np.shape(s_val))
	print('Test data have shape', np.shape(x_test), np.shape(t_test), np.shape(s_test))

	utc = datetime.datetime.utcnow().strftime('%s')

	n_outputs = 2

	results_fn = os.path.join(
		os.path.split(os.getcwd())[0],
		'results',
		'synthetic_baselines_' + utc + '.csv')

	head = ['status', 'model_type', 'hidden_layer_size',
			'n_iter', 'final_train_nll', 'final_val_nll']
	head += ['mean_auc'] + [('auc%i' % i) for i in range (n_outputs)]
	head += ['mean_raem'] + [('raem%i' % i) for i in range(n_outputs)]
	head += ['mean_raea'] + [('raea%i' % i) for i in range(n_outputs)]
	head += ['mean_ci'] + [('ci%i' % i) for i in range(n_outputs)]

	with open(results_fn, 'w+') as results_file:
		print(', '.join(head), file=results_file)

	for i in range(3 * 10):

		hidden_layer_sizes = (100, )
		model_type = ['survival', 'c_mlp', 's_mlp'][i % 3]

		if model_type is 'survival':
			train_data = [x_train, c_train, t_train, s_train]
			val_data = [x_val, c_val, t_val, s_val]
		elif model_type is 'c_mlp':
			train_data = [x_train, c_train]
			val_data = [x_val, c_val]
		else:
			train_data = [x_train, s_train]
			val_data = [x_val, s_val]

		test_data = [x_test, c_test, t_test, s_test]

		print('Running', model_type, 'with layers', hidden_layer_sizes)

		#try:

		n_iter, final_train_nll, final_val_nll, aucs, raes_median, raes_all, cis = train_baseline(
			model_type, hidden_layer_sizes,
			train_data,
			val_data,
			test_data)
		status = 'complete'

		# except:
		# 	n_iter, final_train_nll, final_val_nll = [np.nan] * 3
		# 	aucs = [np.nan] * n_outputs
		# 	raes_median = [np.nan] * n_outputs
		# 	raes_all = [np.nan] * n_outputs
		# 	cis = [np.nan] * n_outputs
		# 	status = 'failed'

		results = [status, model_type, hidden_layer_sizes[0],
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


def train_baseline(
	model_type, hidden_layer_sizes,
	train_data, val_data, test_data):

	tf.reset_default_graph()

	if model_type is 'survival':
		mdl = SurvivalModel(decoder_layer_sizes=hidden_layer_sizes, dropout_pct=.5)
	else:
		mdl = PosNegModel(encoder_layer_sizes=hidden_layer_sizes, dropout_pct=.5)

	with tf.Session() as sess:
		train_stats, val_stats = mdl.train(
			sess, train_data, val_data,
			100, max_epochs_no_improve=3, learning_rate=3e-4,
			verbose=False)
		predictions = mdl.predict(sess, test_data[0])
		
	train_stats = list(zip(*train_stats))
	val_stats = list(zip(*val_stats))

	n_iter = train_stats[0][-1]
	final_train_nll = train_stats[1][-1]
	final_val_nll = val_stats[1][-1]

	x_test, c_test, t_test, s_test = test_data
	n_out = np.shape(s_test)[1]

	if model_type is 'survival':
		c_prob_pred = t_to_prob(predictions)
		raes = [rae_over_samples(t_test[:, i], s_test[:, i], predictions[..., i]) for i in range(n_out)]
		print('raes are', raes)
		cis = [ci(t_test[:, i], s_test[:, i], predictions[..., i]) for i in range(n_out)]
		raes_median, raes_all = list(zip(*raes))
	else:
		c_prob_pred = predictions
		raes_median = [np.nan] * n_out
		raes_all = [np.nan] * n_out
		cis = [np.nan] * n_out

	aucs = [roc_auc_score(c_test[:, i], c_prob_pred[:, i]) for i in range(n_out)]

	return n_iter, final_train_nll, final_val_nll, aucs, list(raes_median), list(raes_all), cis


def time_to_prob(x):
	sorted_pos = {v: pos for pos, v in enumerate(sorted(x))}
	return 1 - np.array([sorted_pos[v] / len(x) for v in x])

def t_to_prob(x):
	return np.stack([time_to_prob(arr) for arr in x.T]).T


class SurvivalModel:

	def __init__(self,
		decoder_layer_sizes=(),
		dropout_pct=0.,
		activation_fn=tf.nn.relu):

		self.decoder_layer_sizes = decoder_layer_sizes
		self.dropout_pct = dropout_pct
		self.activation_fn = activation_fn


	def train(self, sess,
			  train_data,
			  val_data,
			  max_epochs, max_epochs_no_improve=0,
			  learning_rate=1e-3,
			  batch_size=400, batch_eval_freq=10,
			  verbose=False):

		self.n_features = np.shape(train_data[0])[1]
		self.n_outputs = np.shape(train_data[1])[1]

		self.max_t = np.amax(train_data[2])
		### NOTE ON MAX T: this may be breaking the non-fpr version ###
		### Should be dimension-specific ###

		self.opt = tf.train.AdamOptimizer(learning_rate)
		self._build_placeholders()
		self._build_model()

		sess.run(tf.global_variables_initializer())

		train_stats = []
		val_stats = []
		best_val_nloglik = np.inf
		n_epochs_no_improve = 0

		batches_per_epoch = int(np.ceil(
			np.shape(train_data[0])[0] / batch_size))

		for epoch_idx in range(max_epochs):

			for batch_idx, (xb, cb, tb, sb) in enumerate(get_batch(
				train_data, batch_size)):

				lgnrm_nlogp_, lgnrm_nlogs_, _ = sess.run(
					[self.lgnrm_nlogp, self.lgnrm_nlogs, self.train_op],
					feed_dict={self.x: xb, self.t: tb, self.s: sb, self.is_training: True})

				if np.isnan(np.mean(lgnrm_nlogp_)):
					print('Warning: lgnrm_nlogp is NaN')
				if np.isnan(np.mean(lgnrm_nlogs_)):
					print('Warning: lgnrm_nlogs is NaN')

				if batch_idx % batch_eval_freq == 0:
					idx = epoch_idx * batches_per_epoch + batch_idx
					train_stats.append(
						(idx, ) + self._get_train_stats(
							sess, xb, tb, sb))

			idx = (epoch_idx + 1) * batches_per_epoch
			val_stats.append(
				(idx, ) + self._get_train_stats(
				sess, val_data[0], val_data[2], val_data[3]))

			if epoch_idx % 10 == 0:

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
		
		self.x = tf.placeholder(
			shape=(None, self.n_features),
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


	def _get_train_stats(self, sess, xs, ts, ss):

		nloglik, mean, logvar = sess.run(
			[self.nll,
			 self.t_mu,
			 self.t_logvar],
			feed_dict={self.x: xs, self.t: ts, self.s:ss, self.is_training: False})

		return nloglik, np.mean(mean), np.mean(logvar)


	def _summarize(self, train_stats, val_stats, batches_per_epoch):

		print('nloglik (train) = %.2e' % train_stats[1])
		print('t_mu: %.2e' % train_stats[2], 't_logvar: %.2e' % train_stats[3])
		print('nloglik (val) = %.2e' % val_stats[1])
		print('t_mu: %.2e' % val_stats[2], 't_logvar: %.2e\n' % val_stats[3])


	def predict(self, sess, x_test):

		return sess.run(
			self.t_pred,
			feed_dict={self.x: x_test, self.is_training: False})


class PosNegModel:

	def __init__(self, encoder_layer_sizes=(),
		dropout_pct=0.,
		activation_fn=tf.nn.relu):

		self.encoder_layer_sizes = encoder_layer_sizes
		self.dropout_pct = dropout_pct
		self.activation_fn = activation_fn


	def train(self, sess,
			  train_data,
			  val_data,
			  max_epochs, max_epochs_no_improve=0,
			  learning_rate=1e-3,
			  batch_size=300, batch_eval_freq=10,
			  verbose=False):

		self.n_features = np.shape(train_data[0])[1]
		self.n_outputs = np.shape(train_data[1])[1]

		self.opt = tf.train.AdamOptimizer(learning_rate)
		self._build_placeholders()
		self._build_model()

		sess.run(tf.global_variables_initializer())

		train_stats = []
		val_stats = []
		best_val_nloglik = np.inf
		n_epochs_no_improve = 0

		batches_per_epoch = int(np.ceil(
			np.shape(train_data[0])[0] / batch_size))

		for epoch_idx in range(max_epochs):

			for batch_idx, (xb, sb) in enumerate(
				get_batch(train_data, batch_size)):

				sess.run(
					self.train_op,
					feed_dict={self.x: xb, self.s: sb, self.is_training: True})

				if batch_idx % batch_eval_freq == 0:
					idx = epoch_idx * batches_per_epoch + batch_idx
					train_stats.append((idx, self._get_train_stats(sess, xb, sb)))

			idx = (epoch_idx + 1) * batches_per_epoch
			val_stats.append(
				(idx, self._get_train_stats(sess, val_data[0], val_data[1])))

			if epoch_idx % 10 == 0:

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
		
		self.x = tf.placeholder(
			shape=(None, self.n_features),
			dtype=tf.float32)

		self.s = tf.placeholder(
			shape=(None, self.n_outputs),
			dtype=tf.float32)

		self.is_training = tf.placeholder(
			shape=(),
			dtype=tf.bool)


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


	def _get_train_stats(self, sess, xs, ss):

		nloglik = sess.run(
			self.nll,
			feed_dict={self.x: xs, self.s:ss, self.is_training: False})

		return nloglik


	def _summarize(self, train_stats, val_stats, batches_per_epoch):

		print('nloglik (train) = %.2e' % train_stats)
		print('nloglik (val) = %.2e' % val_stats)


	def predict(self, sess, x_test):

		return sess.run(self.s_probs, feed_dict={self.x: x_test, self.is_training: False})


def get_batch(arrs, n=1):
    l = len(arrs[0])
    for ndx in range(0, l, n):
        yield (arr[ndx:min(ndx + n, l)] for arr in arrs)


if __name__ == '__main__':
	main()

