import numpy as np
import tensorflow as tf

LOG_SQRT_2PI = np.log(np.sqrt(2 * np.pi))
PENALTY = 1e10


class CFTModel:

	def __init__(self, c_layer_sizes=(), f_layer_sizes=()):

		self.c_layer_sizes = c_layer_sizes
		self.f_layer_sizes = f_layer_sizes


	def train(self, sess, x_train, t_train, s_train,
			  n_epochs, learning_rate=1e-3, batch_size=300):

		assert np.shape(t_train)[1] == np.shape(s_train)[1]

		self.n_features = np.shape(x_train)[1]
		self.n_outputs = np.shape(t_train)[1]

		self.max_t = np.amax(t_train)

		self._build_placeholders()
		self._build_c_model()
		self._build_t_model()
		self._build_nloglik()

		train_step = tf.train.AdamOptimizer(
			learning_rate).minimize(self.nloglik)

		sess.run(tf.global_variables_initializer())

		print('Initial Values:')
		self._summarize(sess, x_train, t_train, s_train)

		for epoch in range(n_epochs):

			for x_batch, t_batch, s_batch in get_batch(
				batch_size, x_train, t_train, s_train):

				sess.run(train_step,
					feed_dict={self.x: x_batch, self.t: t_batch, self.s:s_batch})

			if epoch % 10 == 0:

				print('Completed Epoch %i' % epoch)
				self._summarize(sess, x_train, t_train, s_train)


	def _build_placeholders(self):
		
		self.x = tf.placeholder(
			shape=(None, self.n_features),
			dtype=tf.float64)

		self.t = tf.placeholder(
			shape=(None, self.n_outputs),
			dtype=tf.float64)

		self.s = tf.placeholder(
			shape=(None, self.n_outputs),
			dtype=tf.float64)


	def _build_c_model(self):

		hidden_layer = self.x

		for layer_size in self.c_layer_sizes:

			hidden_layer = tf.layers.dense(
				hidden_layer, layer_size,
				activation=tf.nn.elu)

		self.c_logits = tf.layers.dense(
			hidden_layer, self.n_outputs,
			activation=None, name='c_logit_weights')

		self.c_logit_weights = tf.get_default_graph().get_tensor_by_name(
			'c_logit_weights/kernel:0')

		self.c_probs = tf.nn.sigmoid(self.c_logits)


	def _build_t_model(self):

		hidden_layer = self.x

		for layer_size in self.f_layer_sizes:

			hidden_layer = tf.layers.dense(
				hidden_layer, layer_size,
				activation=tf.nn.elu)

		self.t_mu = tf.layers.dense(
			hidden_layer, self.n_outputs,
			activation=None)

		#self.t_sig = tf.exp(tf.layers.dense(
		#	hidden_layer, self.n_outputs,
		#	activation=None))

		self.t_sig = tf.constant(0.5, dtype=tf.float64)

		self.t_pred = tf.exp(self.t_mu + tf.random_normal(
			shape=tf.shape(self.t_sig),
			dtype=tf.float64) * self.t_sig)


	def _build_nloglik(self):

		# NOTE: n_outputs > 1 not yet implemented

		self.log_p_ts_given_c_is_1 = self.s * lognormal_logpdf(
			self.t, self.t_mu, self.t_sig)

		self.log_p_ts_given_c_is_1 += self.s * log_uniform_survival(
			self.t, self.max_t * 1.1)

		self.log_p_ts_given_c_is_1 += (1 - self.s) * lognormal_logsurvival(
			self.t, self.t_mu, self.t_sig)

		self.log_p_ts_given_c_is_0 = self.s * -1 * PENALTY

		nloglik = nlog_sigmoid(self.c_logits) * self.log_p_ts_given_c_is_1
		nloglik += nlog_sigmoid(-1 * self.c_logits) * self.log_p_ts_given_c_is_0

		self.nloglik = tf.reduce_mean(nloglik)


	def _summarize(self, sess, xs, ts, ss):

		nloglik, c1, c0, logsig, logmean = sess.run(
			[self.nloglik,
			 self.log_p_ts_given_c_is_1,
			 self.log_p_ts_given_c_is_0,
			 self.t_sig,
			 self.t_mu],
			feed_dict={self.x: xs, self.t: ts, self.s:ss})

		print('nloglik = %.2e' % nloglik)
		print('log_p(t, s | x, c = 1): %.2e' % np.mean(c1))
		print('log_p(t, s | x, c = 0): %.2e' % np.mean(c0))
		print('t_mu: %.2e' % np.mean(logmean), 't_sig: %.2e\n' % np.mean(logsig))


	def predict_c(self, sess, x_test):

		return sess.run(self.c_probs, feed_dict={self.x: x_test})


	def predict_t(self, sess, x_test):

		return sess.run(self.t_pred, feed_dict={self.x: x_test})


	def get_c_weights(self, sess):

		return sess.run(self.c_logit_weights)


def lognormal_logpdf(t, mu, sig):

	logt = tf.log(t)

	normal_dist = tf.distributions.Normal(
		loc=mu, scale=sig)
	
	return normal_dist.log_prob(logt) - logt


def lognormal_logsurvival(t, mu, sig):

	normal_dist = tf.distributions.Normal(
		loc=mu, scale=sig)
	
	return normal_dist.log_survival_function(tf.log(t))


def nlog_sigmoid(logits):
	return tf.log(tf.exp(-1 * logits) + 1)


def log_uniform_survival(t, max_t):
	return 1 - t / max_t


def get_batch(batch_size, *arrs):
    combined = list(zip(*arrs))
    l = len(combined)
    for ndx in range(0, l, batch_size):
        yield zip(*combined[ndx:min(ndx + batch_size, l)])
