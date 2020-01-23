import numpy as np
import tensorflow as tf
from tensorflow.python.ops.parallel_for.gradients import jacobian

EPS = 1e-4
PENALTY = 4


class CFTModel:

	def __init__(self, encoder_layer_sizes=(),
		decoder_layer_sizes=(),
		dropout_pct=0., estimator='none',
		n_samples=30,# gs_temperature=1e-1,
		fpr_likelihood=False,
		activation_fn=tf.nn.relu):

		self.encoder_layer_sizes = encoder_layer_sizes
		self.decoder_layer_sizes = decoder_layer_sizes
		self.dropout_pct = dropout_pct
		self.estimator = estimator
		self.n_samples = n_samples
		#self.gs_temperature = gs_temperature
		self.fpr_likelihood = fpr_likelihood
		self.activation_fn = activation_fn


	def train(self, sess,
			  x_train, t_train, s_train,
			  x_val, t_val, s_val,
			  max_epochs, max_epochs_no_improve=0,
			  learning_rate=1e-3,
			  batch_size=300, batch_eval_freq=10,
			  verbose=False):

		self.n_features = np.shape(x_train)[1]
		self.n_outputs = np.shape(t_train)[1]

		assert np.shape(x_train)[1] == self.n_features
		assert np.shape(s_train)[1] == self.n_outputs
		assert np.shape(t_val)[1] == self.n_outputs
		assert np.shape(s_val)[1] == self.n_outputs

		# check if this is the problem -- might be taking log 0 later

		self.max_t = np.amax(t_train)
		### NOTE ON MAX T: may want to make this dimension-specific ###

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

		batches_per_epoch = int(np.ceil(
			np.shape(x_train)[0] / batch_size))

		for epoch_idx in range(max_epochs):

			for batch_idx, (xb, tb, sb) in enumerate(get_batch(
				batch_size, x_train, t_train, s_train)):

				lgnrm_nlogp_, lgnrm_nlogs_, unif_nlogs_, _ = sess.run(
					[self.lgnrm_nlogp, self.lgnrm_nlogs, self.unif_nlogs, self.train_op],
					feed_dict={self.x: xb, self.t: tb, self.s: sb})

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
			val_stats.append(
				(idx, ) + self._get_train_stats(
				sess, x_val, t_val, s_val))

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


	def _build_model_no_estimator(self):

		self.c_logits, self.c_probs = self._encoder(self.x)
		self.t_mu, self.t_logvar = self._decoder(self.x)

		nll = self._nloglik(self.c_probs, self.t_mu, self.t_logvar)
		self.nll = tf.reduce_mean(nll)

		self.train_op = self.opt.minimize(self.nll)


	def _build_model_gs_estimator(self):

		self._build_p_given_c()

		c = sample_gumbel_softmax(
			self.c_logits,
			self.gs_temperature,
			self.n_gs_samples)

		self.nloglik = tf.reduce_mean(self._nloglik(c))


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

			#fp_nlogprob = -1 * np.log(1e-1)
			fp_nlogp = nlog_sigmoid(self.c_logits)

			if not self.estimator == 'none':
				fp_nlogp = fp_nlogp[:, tf.newaxis, :]

			nll = s * self.lgnrm_nlogp
			nll += (1 - s) * c * self.lgnrm_nlogs
			nll += s * (1 - c) * fp_nlogp

		else:

			p_c1 = s * (self.lgnrm_nlogp + self.unif_nlogs)
			p_c1 += (1 - s) * self.lgnrm_nlogs
			p_c0 = s * PENALTY

			nll = p_c1 * c + p_c0 * (1 - c)

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


	def predict_c(self, sess, x_test):

		return sess.run(self.c_probs, feed_dict={self.x: x_test})


	def predict_t(self, sess, x_test):

		return sess.run(self.t_pred, feed_dict={self.x: x_test})


	def get_c_weights(self, sess):

		return sess.run(self.c_logit_weights)


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


def lognormal_nlogpdf(t, mu, logvar):

	logt = tf.log(t + EPS)
	scale = tf.exp(0.5 * logvar)

	normal_dist = tf.distributions.Normal(
		loc=mu, scale=scale)
	
	return logt - normal_dist.log_prob(logt)


def lognormal_nlogsurvival(t, mu, logvar):

	logt = tf.log(t + EPS)
	scale = tf.exp(0.5 * logvar)

	normal_dist = tf.distributions.Normal(
		loc=mu, scale=scale)
	
	return -1 * normal_dist.log_survival_function(logt)


def uniform_nlogsurvival(t, max_t):
	return -1 * tf.log(1 + EPS - t / max_t)


def nlog_sigmoid(logits):
	return tf.log(tf.exp(-1 * logits) + 1)


def get_batch(batch_size, *arrs):
	combined = list(zip(*arrs))
	l = len(combined)
	for ndx in range(0, l, batch_size):
		yield zip(*combined[ndx:min(ndx + batch_size, l)])


def sample_gumbel(shape, eps=1e-10):
	"""Sample from Gumbel(0, 1)"""
	U = tf.random_uniform(shape, minval=0, maxval=1)
	return -tf.log(-tf.log(U + eps) + eps)


def sample_gumbel_softmax(logits, temperature, n_samples=1):
	""" Draw samples from the Gumbel-Softmax distribution"""
	samples = sample_gumbel((n_samples, ))
	y = logits[:, :, tf.newaxis] + samples[tf.newaxis, tf.newaxis, :]
	return tf.nn.sigmoid(y / temperature)  # check dimension


def tile_and_flatten(x, n_samples, n_features):
	xt = tf.tile(x[:, tf.newaxis, :], (1, n_samples, 1))
	return flatten_samples(xt, n_features)


def flatten_samples(x, n_features):
	return tf.reshape(x, (-1, n_features))


def unflatten_samples(x, n_samples, n_features):
	return tf.reshape(x, (-1, n_samples, n_features))


