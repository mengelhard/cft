import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import os
from datetime import datetime

MIMIC_DIR = '/Users/mme/projects/cft/data/mimic'
#MIMIC_DIR = '/scratch/mme4/mimic/batches'

TIME_FMT = '%Y-%m-%d %H:%M:%S'


class CFTModelMimic:

	def __init__(self,
		encoder_layer_sizes=(),
		decoder_layer_sizes=(),
		dropout_pct=0., estimator='none',
		n_samples=30, gs_temperature=1.,
		fpr=5e-2,
		fpr_likelihood=False,
		prop_fpr=True,
		activation_fn=tf.nn.relu):

		self.encoder_layer_sizes = encoder_layer_sizes
		self.decoder_layer_sizes = decoder_layer_sizes
		self.dropout_pct = dropout_pct
		self.estimator = estimator
		self.n_samples = n_samples
		self.gs_temperature = gs_temperature
		self.nlog_fpr = -1 * np.log(fpr)
		self.fpr_likelihood = fpr_likelihood
		self.prop_fpr = prop_fpr
		self.activation_fn = activation_fn


	def train(self, sess,
			  train_file_indices,
			  val_file_indices,
			  max_epochs, max_epochs_no_improve=1,
			  learning_rate=1e-3,
			  batch_size=300, batch_eval_freq=1,
			  verbose=False):

		self.n_outputs = 10
		self.n_features = 'None'
		self.opt = tf.train.AdamOptimizer(learning_rate)
		self._build_placeholders()

		if self.estimator == 'arm':
			self._build_model_arm_estimator(self.n_samples)
		elif self.estimator == 'gs':
			self._build_model_gs_estimator(self.n_samples)
		else:
			self._build_model_no_estimator()

		sess.run(tf.global_variables_initializer())

		train_stats = []
		val_stats = []
		best_val_nloglik = np.inf
		n_epochs_no_improve = 0

		batches_per_epoch = len(train_file_indices)

		event_dict = load_pickle(os.path.join(MIMIC_DIR, 'events.pickle'))
		feature_dict = load_pickle(os.path.join(MIMIC_DIR, 'itemid_dict.pickle'))

		for epoch_idx in range(max_epochs):

			for batch_idx, batch_file_idx in enumerate(train_file_indices):

				xb, cb, tb, sb = load_batch(batch_file_idx, event_dict, feature_dict)

				lgnrm_nlogp_, lgnrm_nlogs_, unif_nlogs_, _ = sess.run(
					[self.lgnrm_nlogp, self.lgnrm_nlogs, self.unif_nlogs, self.train_op],
					feed_dict={self.x: cb, self.t: tb, self.s: sb})

				if np.isnan(np.mean(lgnrm_nlogp_)):
					print('Warning: lgnrm_nlogp is NaN')
				if np.isnan(np.mean(lgnrm_nlogs_)):
					print('Warning: lgnrm_nlogs is NaN')
				if np.isnan(np.mean(unif_nlogs_)):
					print('Warning: unif_nlogs is NaN')

				if batch_idx % batch_eval_freq == 0:
					idx = epoch_idx * batches_per_epoch + batch_idx
					train_stats.append(
						(idx, ) + self._get_train_stats(
							sess, xb, tb, sb))

			idx = (epoch_idx + 1) * batches_per_epoch

			current_val_stats = []

			for val_batch_idx, batch_file_idx in enumerate(val_file_indices):

				xb, cb, tb, sb = load_batch(batch_file_idx, event_dict, feature_dict)

				current_val_stats.append(
					self._get_train_stats(
						sess, xb, tb, sb))

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


	def _build_model_no_estimator(self):

		self.c_logits, self.c_probs = self._encoder(self.x)
		self.t_mu, self.t_logvar = self._decoder(self.x)

		nll = self._nloglik(self.c_probs, self.t_mu, self.t_logvar)
		self.nll = tf.reduce_mean(nll)

		self.train_op = self.opt.minimize(self.nll)


	def _build_model_gs_estimator(self, n_samples):

		self.c_logits, self.c_probs = self._encoder(self.x)

		c = sample_gumbel_bernoulli(
			self.c_logits,
			self.gs_temperature,
			n_samples)

		x = tf.tile(self.x[:, tf.newaxis, :], (1, n_samples, 1))
		xc = tf.concat([x, c], axis=2)

		self.t_mu, self.t_logvar = self._decoder(xc)

		nll = self._nloglik(c, self.t_mu, self.t_logvar)
		self.nll = tf.reduce_mean(nll)

		self.train_op = self.opt.minimize(self.nll)


	def _build_model_arm_estimator(self, n_samples):

		self.c_logits, self.c_probs = self._encoder(self.x)

		unif_samples = tf.random_uniform(
			shape=(1, n_samples, 1),
			dtype=tf.float32)

		# normal c samples

		c = unif_samples < self.c_probs[:, tf.newaxis, :]
		c = tf.cast(c, dtype=tf.float32)

		x = tf.tile(self.x[:, tf.newaxis, :], (1, n_samples, 1))
		xc = tf.concat([x, c], axis=2)

		self.t_mu, self.t_logvar = self._decoder(xc)

		nll = self._nloglik(c, self.t_mu, self.t_logvar)
		self.nll = tf.reduce_mean(nll)

		# calculate gradient wrt decoder

		decoder_grads = self.opt.compute_gradients(
			self.nll, var_list=self.decoder_vars)

		self.decoder_op = self.opt.apply_gradients(decoder_grads)

		# ARM-specific samples

		c_neg = unif_samples > (1 - self.c_probs[:, tf.newaxis, :])
		c_neg = tf.cast(c_neg, dtype=tf.float32)

		xc_neg = tf.concat([x, c_neg], axis=2)

		t_mu_neg, t_logvar_neg = self._decoder(xc_neg, reuse=True)

		nll_neg = self._nloglik(c_neg, t_mu_neg, t_logvar_neg)

		# calculate gradient wrt encoder using ARM

		logit_grads = (nll_neg - nll) * (unif_samples - .5)
		logit_grads = tf.reduce_mean(logit_grads, axis=1)

		encoder_grads = tf.gradients(
			self.c_logits,
			self.encoder_vars,
			grad_ys=logit_grads)

		self.encoder_op = self.opt.apply_gradients(
			zip(encoder_grads, self.encoder_vars))

		with tf.control_dependencies([self.decoder_op, self.encoder_op]):
			self.train_op = tf.no_op()


	def _nloglik(self, c, t_mu, t_logvar):

		if self.estimator == 'none':
			t = self.t
			s = self.s
		else:
			t = self.t[:, tf.newaxis, :]
			s = self.s[:, tf.newaxis, :]

		self.lgnrm_nlogp = lognormal_nlogpdf(t, self.t_mu, self.t_logvar)
		self.lgnrm_nlogs = lognormal_nlogsurvival(t, self.t_mu, self.t_logvar)
		self.unif_nlogs = uniform_nlogsurvival(t, self.max_t)

		if self.fpr_likelihood:

			nlog_fpr = self.nlog_fpr

			if self.prop_fpr:

				nlog_fpr += nlog_sigmoid(self.c_logits)

				if not self.estimator == 'none':
					nlog_fpr = nlog_fpr[:, tf.newaxis, :]

			nll = s * self.lgnrm_nlogp
			nll += (1 - s) * c * self.lgnrm_nlogs
			nll += s * (1 - c) * nlog_fpr

		else:

			p_c1 = s * (self.lgnrm_nlogp + self.unif_nlogs)
			p_c1 += (1 - s) * self.lgnrm_nlogs
			p_c0 = s * self.nlog_fpr

			nll = p_c1 * c + p_c0 * (1 - c)

		return nll


	def _build_placeholders(self):

		self.x = tf.placeholder(
			shape=(None, self.n_features),
			dtype=tf.float32)

		self.t = tf.placeholder(
			shape=(None, self.n_outputs),
			dtype=tf.float32)

		self.max_t = tf.reduce_max(self.t)

		self.s = tf.placeholder(
			shape=(None, self.n_outputs),
			dtype=tf.float32)


	def _encoder(self, h):

		with tf.variable_scope('encoder'):

			hidden_layer = mlp(
				h, self.encoder_layer_sizes,
				dropout_pct=self.dropout_pct,
				activation_fn=self.activation_fn)

			logits = tf.layers.dense(
				hidden_layer, self.n_outputs,
				activation=None, name='logit_weights')

			probs = tf.nn.sigmoid(logits)

		self.encoder_vars = tf.get_collection(
			tf.GraphKeys.GLOBAL_VARIABLES,
			scope='encoder')

		return logits, probs


	def _decoder(self, h, reuse=False):

		with tf.variable_scope('decoder', reuse=reuse):

			hidden_layer = mlp(
				h, self.decoder_layer_sizes,
				dropout_pct=self.dropout_pct,
				activation_fn=self.activation_fn,
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

		self.t_pred = tf.exp(mu + tf.random_normal(
			shape=tf.shape(logvar),
			dtype=tf.float32) * tf.exp(.05 * logvar))

		self.decoder_vars = tf.get_collection(
			tf.GraphKeys.GLOBAL_VARIABLES,
			scope='decoder')

		return mu, logvar


	def _get_train_stats(self, sess, xs, ts, ss):

		nloglik, mean, logvar = sess.run(
			[self.nll,
			 self.t_mu,
			 self.t_logvar],
			feed_dict={self.x: xs, self.t: ts, self.s:ss})

		return nloglik, np.mean(mean), np.mean(logvar)


	def _summarize(self, train_stats, val_stats, batches_per_epoch):

		print('nloglik (train) = %.2e' % train_stats[1])
		print('t_mu: %.2e' % train_stats[2], 't_logvar: %.2e' % train_stats[3])
		print('nloglik (val) = %.2e' % val_stats[1])
		print('t_mu: %.2e' % val_stats[2], 't_logvar: %.2e\n' % val_stats[3])


	def predict_c_and_t(self, sess, filenames):

		c_probs = []
		t_pred = []
		c = []
		t = []
		s = []

		for idx, batch_file in enumerate(filenames):

			xb, cb, tb, sb = load_batch(batch_file)

			c_probs_, t_pred_ = sess.run(
				[self.c_probs, self.t_pred],
				feed_dict={self.x: xb})

			c_probs.append(c_probs_)
			t_pred.append(t_pred_)

			c.append(cb)
			t.append(tb)
			s.append(sb)

		c_probs = np.concatenate(c_probs, axis=0)
		t_pred = np.concatenate(t_pred, axis=0)
		c = np.concatenate(c, axis=0)
		t = np.concatenate(t, axis=0)
		s = np.concatenate(s, axis=0)

		return c_probs, t_pred, c, t, s


def mlp(x, hidden_layer_sizes,
		dropout_pct = 0.,
		activation_fn=tf.nn.relu,
		reuse=False):

	hidden_layer = x

	with tf.variable_scope('mlp', reuse=reuse):

		for i, layer_size in enumerate(hidden_layer_sizes):

			hidden_layer = tf.layers.dense(
				hidden_layer, layer_size,
				activation=activation_fn,
				name='fc_%i' % i,
				reuse=reuse)

			if dropout_pct > 0:
				hidden_layer = tf.layers.dropout(
					hidden_layer, rate=dropout_pct,
					name='dropout_%i' % i)

	return hidden_layer


def lognormal_nlogpdf(t, mu, logvar, epsilon=1e-4):

	logt = tf.log(t + epsilon)
	scale = tf.exp(0.5 * logvar)

	normal_dist = tf.distributions.Normal(
		loc=mu, scale=scale)
	
	return logt - normal_dist.log_prob(logt)


def lognormal_nlogsurvival(t, mu, logvar, epsilon=1e-4):

	logt = tf.log(t + epsilon)
	scale = tf.exp(0.5 * logvar)

	normal_dist = tf.distributions.Normal(
		loc=mu, scale=scale)
	
	return -1 * normal_dist.log_survival_function(logt)


def uniform_nlogsurvival(t, max_t, epsilon=1e-4):
	return -1 * tf.log(1 + epsilon - t / max_t)


def nlog_sigmoid(logits):
	return tf.log(tf.exp(-1 * logits) + 1)


def load_pickle(fn):
	with open(fn, 'rb') as file:
		p = pickle.load(file)
	return p


def get_events(pt):

	lab_event_labels = [
		'WBC, CSF', 'Troponin T', 'Intubated', 'WBC, Pleural',
		'Thyroid Stimulating Hormone', 'D-Dimer', 'Urobilinogen',
		'Anti-Nuclear Antibody', 'Ammonia', 'Lipase']

	start_time = pd.Series(
		[pt['admission_time'], pt['first_chart_measurement'], pt['first_lab_measurement']],
		index=[0, 1, 2]).min()

	times = {l: t for l, n, t in pt['events']}
	event_times = [timediff_hours(start_time, times[l]) for l in lab_event_labels]

	n_occur = {l: n for l, n, t in pt['events']}
	event_occurrence = [(n_occur[l] > 0).astype(float) for l in lab_event_labels]

	et_eo = list(zip(event_times, event_occurrence))
	assert np.shape(et_eo) == (10, 2)

	return et_eo


def timediff_hours(t1, t2):
	td = datetime.strptime(t2, TIME_FMT) - datetime.strptime(t1, TIME_FMT)
	return td.total_seconds() / (60 * 60)


def normalize(arr, epsilon=1e-4):
	a = np.array(arr)
	return (a - a.mean()) / np.sqrt(a.var() + epsilon)


def load_batch(file_idx, edict, fdict, asframe=False, normalize=True):

	chartevents = pd.read_csv(os.path.join(MIMIC_DIR, 'chartevents_%i.csv' % file_idx),
		usecols=['HADM_ID', 'ITEMID', 'CHARTTIME_HOURS', 'VALUENUM'],
		dtype={'HADM_ID': int, 'ITEMID': int, 'CHARTTIME_HOURS': float, 'VALUENUM': float})

	labevents = pd.read_csv(os.path.join(MIMIC_DIR, 'labevents_%i.csv' % file_idx),
		usecols=['HADM_ID', 'ITEMID', 'CHARTTIME_HOURS', 'VALUENUM'],
		dtype={'HADM_ID': int, 'ITEMID': int, 'CHARTTIME_HOURS': float, 'VALUENUM': float})

	outputevents = pd.read_csv(os.path.join(MIMIC_DIR, 'outputevents_%i.csv' % file_idx),
		usecols=['HADM_ID', 'ITEMID', 'CHARTTIME_HOURS', 'VALUENUM'],
		dtype={'HADM_ID': int, 'ITEMID': int, 'CHARTTIME_HOURS': float, 'VALUENUM': float})

	chartfeatures_numeric = get_features(
		chartevents,
		fdict['chartevents_numeric'],
		['count', 'mean', 'min', 'max'],
		normalize=normalize)

	chartfeatures_nonnumeric = get_features(
		chartevents,
		fdict['chartevents_nonnumeric'],
		['count'],
		normalize=normalize)

	labfeatures = get_features(
		labevents,
		fdict['labevents'],
		['count', 'mean', 'min', 'max'],
		normalize=normalize)

	outputfeatures = get_features(
		outputevents,
		fdict['outputevents'],
		['count', 'sum'],
		normalize=normalize)

	features = chartfeatures_numeric.join(
		chartfeatures_nonnumeric, how='left').join(
		labfeatures, how='left').join(
		outputfeatures, how='left')

	features = features.fillna(0.) # for ids with no features e.g. in outputfeatures

	hadm_ids = features.index.values

	events = np.array([get_events(edict[hadm_id]) for hadm_id in hadm_ids])

	t = events[:, :, 0].astype('float') + 1e-2 # pad with .01 hours to avoid zeros
	c = events[:, :, 1].astype('float')

	if t.min() < 0:
		print('Warning: found t value less than zero')

	all_event_times = t.flatten()[c.flatten() == 1]
	simulated_censoring_times = np.random.rand(*np.shape(t)) * all_event_times.median() * 2 + 1e-2

	s = ((t < simulated_censoring_times) & (c == 1)).astype('float')
	t = np.minimum(t, simulated_censoring_times)

	if asframe:
		return features, c, t, s
	else:
		return features.values, c, t, s


def get_features(df, item_ids, feature_types, normalize=True):

	feature_cols = ['%s_%s' % (f, i) for f in feature_types for i in item_ids]

	fdf = df.groupby(['HADM_ID', 'ITEMID'])['VALUENUM'].agg(feature_types).unstack()
	fdf.columns = ['_'.join(str(c) for c in col) for col in fdf.columns.values]

	# fill median values

	fdf = fdf.fillna(fdf.median(axis=0))

	# normalize columns

	if normalize:
		fdf = (fdf - fdf.mean(axis=0)) / (fdf.std(axis=0) + 1e-4)

	# add columns that aren't present and fill with zeros

	for col in feature_cols:
		if col not in fdf.columns:
			fdf[col] = np.nan

	# standardize feature order

	return fdf[feature_cols]


def sample_gumbel(shape, eps=1e-10):
	"""Sample from Gumbel(0, 1)"""
	U = tf.random_uniform(shape, minval=0, maxval=1)
	return -tf.log(-tf.log(U + eps) + eps)


def sample_gumbel_bernoulli(logits, temperature=1., n_samples=1):
	""" Draw samples from the Gumbel-Bernoulli distribution"""
	samples_centered = sample_gumbel((n_samples, )) - sample_gumbel((n_samples, ))
	y = logits[:, tf.newaxis, :] + samples_centered[tf.newaxis, :, tf.newaxis]
	return tf.nn.sigmoid(y / temperature)


def tile_and_flatten(x, n_samples, n_features):
	xt = tf.tile(x[:, tf.newaxis, :], (1, n_samples, 1))
	return flatten_samples(xt, n_features)


def flatten_samples(x, n_features):
	return tf.reshape(x, (-1, n_features))


def unflatten_samples(x, n_samples, n_features):
	return tf.reshape(x, (-1, n_samples, n_features))


