import numpy as np
import tensorflow as tf
import pickle


class CFTModelReddit:

	def __init__(self,
		embedding_layer_sizes=(),
		encoder_layer_sizes=(),
		decoder_layer_sizes=(),
		dropout_pct=0., estimator='none',
		n_samples=30, gs_temperature=1.,
		fpr=5e-2,
		fpr_likelihood=False,
		prop_fpr=True,
		activation_fn=tf.nn.relu):

		self.embedding_layer_sizes = embedding_layer_sizes
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
			  train_files,
			  val_files,
			  max_epochs, max_epochs_no_improve=1,
			  learning_rate=1e-3,
			  batch_size=300, batch_eval_freq=1,
			  verbose=False):

		self.n_outputs = 9
		self.opt = tf.train.AdamOptimizer(learning_rate)
		self._build_placeholders()
		self._build_x()

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

		batches_per_epoch = len(train_files)

		for epoch_idx in range(max_epochs):

			for batch_idx, batch_file in enumerate(train_files):

				xvb, xfb, cb, tb, sb = load_batch(batch_file)

				lgnrm_nlogp_, lgnrm_nlogs_, unif_nlogs_, _ = sess.run(
					[self.lgnrm_nlogp, self.lgnrm_nlogs, self.unif_nlogs, self.train_op],
					feed_dict={self.xv: xvb, self.xf: xfb, self.t: tb, self.s: sb, self.is_training: True})

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
							sess, xvb, xfb, tb, sb))

			idx = (epoch_idx + 1) * batches_per_epoch

			current_val_stats = []

			for val_batch_idx, batch_file in enumerate(val_files):

				xvb, xfb, cb, tb, sb = load_batch(batch_file)

				current_val_stats.append(
					self._get_train_stats(
						sess, xvb, xfb, tb, sb))

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
		
		self.xv = tf.placeholder(
			shape=(None, 20, 512),
			dtype=tf.float32)

		self.xf = tf.placeholder(
			shape=(None, 2),
			dtype=tf.float32)

		self.t = tf.placeholder(
			shape=(None, self.n_outputs),
			dtype=tf.float32)

		self.max_t = tf.reduce_max(self.t)

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

		self.t_pred = tf.exp(mu + tf.random_normal(
			shape=tf.shape(logvar),
			dtype=tf.float32) * tf.exp(.05 * logvar))

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


	def predict_c_and_t(self, sess, filenames):

		c_probs = []
		t_pred = []
		c = []
		t = []
		s = []

		for idx, batch_file in enumerate(filenames):

			xvb, xfb, cb, tb, sb = load_batch(batch_file)

			c_probs_, t_pred_ = sess.run(
				[self.c_probs, self.t_pred],
				feed_dict={self.xv: xvb, self.xf: xfb, self.is_training: False})

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


	def get_c_weights(self, sess):

		return sess.run(self.c_logit_weights)


def mlp(x, hidden_layer_sizes,
		dropout_pct = 0.,
		activation_fn=tf.nn.relu,
		training=True,
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
					training=training,
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


def get_avg_comment_length(user):
	return np.mean([len(x) for x in user['comments']['raw_texts']])


def get_events(user):

	subreddits = ['ADHD', 'Anxiety', 'books', 'depression', 'Fitness',
				  'LifeProTips', 'mentalhealth', 'SuicideWatch', 'worldnews']

	return [user['events'][subreddit] for subreddit in subreddits]


def normalize(arr, epsilon=1e-4):
	a = np.array(arr)
	return (a - a.mean()) / np.sqrt(a.var() + epsilon)


def load_batch(fn):
	
	batch = load_pickle(fn)
	usernames = list(batch.keys())

	comment_embeddings = np.stack(
		batch[uname]['comments']['encoded'] for uname in usernames)
	comment_lengths = normalize(
		[get_avg_comment_length(batch[uname]) for uname in usernames])

	comment_firsttime = np.array(
		[batch[uname]['comments']['times'][0] for uname in usernames])
	comment_lasttime = np.array(
		[batch[uname]['comments']['times'][-1] for uname in usernames])
	comment_timediff = normalize(comment_lasttime - comment_firsttime)

	events = np.array([get_events(batch[uname]) for uname in usernames])

	t = (events[:, :, 0].astype('float') - comment_lasttime[:, np.newaxis])

	if t.min() < 0:
		print('Warning: found t value less than zero')

	t = (t + 60 * 60) / (60 * 60 * 24 * 30) # pad with 1 hour and convert to months

	#print('min t is %.2f and max t is %.2f' % (t.min(), t.max()))
	c = (events[:, :, 1] == 'event_time').astype('float')

	all_event_times = t.flatten()[c.flatten() == 1]
	simulated_censoring_times = np.random.rand(*np.shape(t)) * np.median(
		all_event_times) * 2.5 + 1e-5

	s = ((t < simulated_censoring_times) & (c == 1)).astype('float')
	t = np.minimum(t, simulated_censoring_times)
	
	x_variable_length = comment_embeddings
	x_fixed_length = np.stack([comment_lengths, comment_timediff]).T
	
	return x_variable_length, x_fixed_length, c, t, s


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


