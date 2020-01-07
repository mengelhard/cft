import numpy as np
import tensorflow as tf

EPS = 1e-8


class CFTModel:

	def __init__(self, c_layer_sizes=(), t_layer_sizes=(),
		dropout_pct=0., arm_estimator=True,
		activation_fn=tf.nn.relu):

		self.c_layer_sizes = c_layer_sizes
		self.t_layer_sizes = t_layer_sizes
		self.dropout_pct = dropout_pct
		self.arm_estimator = arm_estimator
		self.activation_fn = activation_fn


	def train(self, sess,
			  x_train, t_train, s_train,
			  x_val, t_val, s_val,
			  max_epochs, max_epochs_no_improve=0,
			  arm_estimator=True, learning_rate=1e-3,
			  batch_size=300, batch_eval_freq=10):

		self.n_features = np.shape(x_train)[1]
		self.n_outputs = np.shape(t_train)[1]

		assert np.shape(x_train)[1] == self.n_features
		assert np.shape(s_train)[1] == self.n_outputs
		assert np.shape(t_val)[1] == self.n_outputs
		assert np.shape(s_val)[1] == self.n_outputs

		# check if this is the problem -- might be taking log 0 later

		self.max_t = np.amax(t_train)

		self._build_placeholders()
		self._build_c_model()
		self._build_t_model()
		self._build_nloglik()

		train_step = tf.train.AdamOptimizer(
			learning_rate).minimize(self.nloglik)

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

				sess.run(train_step, feed_dict={
					self.x: xb,
					self.t: tb,
					self.s: sb})

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

		hidden_layer = mlp(
			self.x, self.c_layer_sizes,
			dropout_pct=self.dropout_pct,
			activation_fn=self.activation_fn)

		self.c_logits = tf.layers.dense(
			hidden_layer, self.n_outputs,
			activation=None, name='c_logit_weights')

		self.c_logit_weights = tf.get_default_graph().get_tensor_by_name(
			'c_logit_weights/kernel:0')

		self.c_probs = tf.nn.sigmoid(self.c_logits)


	def _build_t_model(self):

		hidden_layer = mlp(
			self.x, self.t_layer_sizes,
			dropout_pct=self.dropout_pct,
			activation_fn=self.activation_fn)

		self.t_mu = tf.layers.dense(
			hidden_layer, self.n_outputs,
			activation=None)

		self.t_logvar = tf.layers.dense(
			hidden_layer, self.n_outputs,
			activation=None)

		#self.t_logvar = tf.constant(np.log(.25), dtype=tf.float64)

		self.t_pred = tf.exp(self.t_mu + tf.random_normal(
			shape=tf.shape(self.t_logvar),
			dtype=tf.float64) * tf.exp(.05 * self.t_logvar))


	def _build_nloglik(self):

		# NOTE: n_outputs > 1 not yet implemented

		fp_nlogprob = .1

		# is this what we want to do??
		fp_nlogprob += nlog_sigmoid(self.c_logits)

		c_prob = tf.sigmoid(self.c_logits)

		nloglik = self.s * lognormal_nlogpdf(
			self.t, self.t_mu, self.t_logvar)

		nloglik += (1 - self.s) * lognormal_nlogsurvival(
			self.t, self.t_mu, self.t_logvar) * c_prob

		nloglik += self.s * fp_nlogprob * (1 - c_prob)

		self.nloglik = tf.reduce_mean(nloglik)


	def _get_train_stats(self, sess, xs, ts, ss):

		nloglik, logvar, mean = sess.run(
			[self.nloglik,
			 self.t_mu,
			 self.t_logvar],
			feed_dict={self.x: xs, self.t: ts, self.s:ss})

		avg_mean = np.mean(mean)
		avg_logvar = np.mean(logvar)

		return nloglik, avg_mean, avg_logvar


	def _summarize(self, train_stats, val_stats, batches_per_epoch):

		print('nloglik (train) = %.2e' % train_stats[1])
		print('t_mu: %.2e' % train_stats[2], 't_logvar: %.2e\n' % train_stats[3])
		print('nloglik (val) = %.2e' % val_stats[1])
		print('t_mu: %.2e' % val_stats[2], 't_logvar: %.2e\n' % val_stats[3])


	def predict_c(self, sess, x_test):

		return sess.run(self.c_probs, feed_dict={self.x: x_test})


	def predict_t(self, sess, x_test):

		return sess.run(self.t_pred, feed_dict={self.x: x_test})


	def get_c_weights(self, sess):

		return sess.run(self.c_logit_weights)


def mlp(x, hidden_layer_sizes,
		dropout_pct = 0., activation_fn=tf.nn.relu):

	hidden_layer = x

	for layer_size in hidden_layer_sizes:

		hidden_layer = tf.layers.dense(
			hidden_layer, layer_size,
			activation=activation_fn)

		if dropout_pct > 0:
			hidden_layer = tf.layers.dropout(
				hidden_layer, rate=dropout_pct)

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

