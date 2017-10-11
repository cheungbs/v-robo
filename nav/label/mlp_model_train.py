import tensorflow as tf
import numpy as np
import os
import argparse

def load_data(data_file):
	data   = np.load(data_file + '.npy')
	data   = data[()]
	feats  = data['feats'].reshape([data['feats'].shape[0], -1])
	labels = data['labels']

	return feats, np.array(labels, dtype=np.int32)

def data_iterator_create(feats_cache, labels_cache, batch_size):
	batch_idx = 0

	while True:
		idxs = np.arange(0, feats_cache.shape[0])
		np.random.shuffle(idxs)
		# shuf_feats  = feats_cache[idxs]
		# shuf_labels = labels_cache[idxs]

		for batch_idx in range(0, len(idxs), batch_size):
			idx_batch    = idxs[batch_idx:batch_idx + batch_size]

			feats_batch  = feats_cache[idx_batch]
			labels_batch = labels_cache[idx_batch]

			yield feats_batch, labels_batch

def train_model(data_file, epochs, save_name, learning_rate=0.001):
	feats_cache, labels_cache = load_data(data_file)
	data_tot, feat_dim  = feats_cache.shape
	label_dim = 512
	num_units = 1024

	feat_in_h = tf.placeholder(tf.float32, [None, feat_dim])
	lbl_out_h = tf.placeholder(tf.int32, [None])

	logits = tf.layers.dense(feat_in_h, num_units, activation=tf.nn.relu)
	logits = tf.layers.dense(logits, num_units, activation=tf.nn.relu)
	logits = tf.layers.dense(logits, label_dim)
	outputs = tf.nn.softmax(logits)

	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=lbl_out_h)
	loss = tf.reduce_mean(loss)

	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	training_op = optimizer.minimize(loss)

	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

	batch_size = 32
	batch_loop = data_tot // batch_size

	sess = tf.Session()
	sess.run(init)
	for ep in range(epochs):
		data_iter = data_iterator_create(feats_cache, labels_cache, batch_size)

		for ib in range(batch_loop):
			feat_b, label_b = data_iter.next()
			_, loss_v = sess.run([training_op, loss], feed_dict={feat_in_h: feat_b, lbl_out_h: label_b})

			if ib % 5 == 0:
				print("epoch: " + str(ep) + " batch: " + str(ib))
				print("loss: " + str(loss_v))

	if not save_name is None:
		save_path = saver.save(sess, save_name)
		print('Model saved in file: %s' % save_path)
	print('The training process is finished.')

	return sess, 


if __name__ == '__main__':

	print('training model.')

	parser = argparse.ArgumentParser(description='Labeling, Resuming, Checking mp4 lables.')
	parser.add_argument('--path'  , default='./', help='video file directory.')
	parser.add_argument('--name'  , default='20170930-064841_0', help='video file name.')

	args = parser.parse_args()

	data_file = os.path.join(args.path, 'vgg16_label_' + args.name)
	epochs    = 100
	mpath     = './models/mlp/model'
	train_model(data_file, epochs, mpath)

