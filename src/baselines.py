import numpy as np
import tensorflow as tf

EPS = 1e-4
PENALTY = 4


class SurvivalModel:

	def __init__(self,
		decoder_layer_sizes=(),
		dropout_pct=0.,
		activation_fn=tf.nn.relu):

		self.decoder_layer_sizes = decoder_layer_sizes
		self.dropout_pct = dropout_pct
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
		self._build_model()

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

				lgnrm_nlogp_, lgnrm_nlogs_, _ = sess.run(
					[self.lgnrm_nlogp, self.lgnrm_nlogs, self.train_op],
					feed_dict={self.x: xb, self.t: tb, self.s: sb})

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
			feed_dict={self.x: xs, self.t: ts, self.s:ss})

		return nloglik, np.mean(mean), np.mean(logvar)


	def _summarize(self, train_stats, val_stats, batches_per_epoch):

		print('nloglik (train) = %.2e' % train_stats[1])
		print('t_mu: %.2e' % train_stats[2], 't_logvar: %.2e' % train_stats[3])
		print('nloglik (val) = %.2e' % val_stats[1])
		print('t_mu: %.2e' % val_stats[2], 't_logvar: %.2e\n' % val_stats[3])


	def predict_t(self, sess, x_test):

		return sess.run(self.t_pred, feed_dict={self.x: x_test})


class PosNegModel:

	def __init__(self, encoder_layer_sizes=(),
		dropout_pct=0.,
		activation_fn=tf.nn.relu):

		self.encoder_layer_sizes = encoder_layer_sizes
		self.dropout_pct = dropout_pct
		self.activation_fn = activation_fn


	def train(self, sess,
			  x_train, s_train,
			  x_val, s_val,
			  max_epochs, max_epochs_no_improve=0,
			  learning_rate=1e-3,
			  batch_size=300, batch_eval_freq=10,
			  verbose=False):

		self.n_features = np.shape(x_train)[1]
		self.n_outputs = np.shape(s_train)[1]

		assert np.shape(x_train)[1] == self.n_features
		assert np.shape(s_train)[1] == self.n_outputs
		assert np.shape(s_val)[1] == self.n_outputs

		self.opt = tf.train.AdamOptimizer(learning_rate)
		self._build_placeholders()
		self._build_model()

		sess.run(tf.global_variables_initializer())

		train_stats = []
		val_stats = []
		best_val_nloglik = np.inf
		n_epochs_no_improve = 0

		batches_per_epoch = int(np.ceil(
			np.shape(x_train)[0] / batch_size))

		for epoch_idx in range(max_epochs):

			for batch_idx, (xb, sb) in enumerate(get_batch(
				batch_size, x_train, s_train)):

				sess.run(self.train_op, feed_dict={self.x: xb, self.s: sb})

				if batch_idx % batch_eval_freq == 0:
					idx = epoch_idx * batches_per_epoch + batch_idx
					train_stats.append((idx, self._get_train_stats(sess, xb, sb)))

			idx = (epoch_idx + 1) * batches_per_epoch
			val_stats.append(
				(idx, self._get_train_stats(sess, x_val, s_val)))

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


	def _get_train_stats(self, sess, xs, ss):

		nloglik = sess.run(
			self.nll,
			feed_dict={self.x: xs, self.s:ss})

		return nloglik


	def _summarize(self, train_stats, val_stats, batches_per_epoch):

		print('nloglik (train) = %.2e' % train_stats)
		print('nloglik (val) = %.2e' % val_stats)


	def predict_prob(self, sess, x_test):

		return sess.run(self.s_probs, feed_dict={self.x: x_test})


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


def get_batch(batch_size, *arrs):
	combined = list(zip(*arrs))
	l = len(combined)
	for ndx in range(0, l, batch_size):
		yield zip(*combined[ndx:min(ndx + batch_size, l)])


