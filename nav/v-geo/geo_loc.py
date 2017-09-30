import tensorflow as tf
import numpy as np

class geo_loc(object):

  def __init__(self, config):
    super(geo_loc, self).__init__()

    self.config = config
    self.X_dim  = self.config.vgg16_dim
    self.y_dim  = self.config.loc_num
    self.len    = np.ones(shape=(1), dtype=np.int32)

    self.num_units = self.config.loc_nn_units

    self.graph = tf.Graph()
    with self.graph.as_default():
      self.X_in_h             = tf.placeholder(tf.float32, [None, None, self.X_dim])
      self.seq_len_h          = tf.placeholder(tf.float32, [None])
      self.init_state_h       = tf.placeholder(tf.float32, [None, self.num_units])
    
      gru_cell                = tf.contrib.rnn.GRUCell(self.num_units)
      rnn_output, self.states = tf.nn.dynamic_rnn(gru_cell, self.X_in_h, dtype=tf.float32, sequence_length=self.seq_len_h, initial_state=self.init_state_h)

      stacked_rnn_output      = tf.reshape(rnn_output, [-1, self.num_units])
      logits                  = tf.layers.dense(stacked_rnn_output, self.y_dim)
      self.outputs            = tf.nn.softmax(logits)

      init = tf.global_variables_initializer()
      saver = tf.train.Saver()

      self.sess = tf.Session()

      self.sess.run(init)
      print("start loading model.")
      saver.restore(self.sess, self.config.geo_loc_path)
      print("model loaded, start to predict.")
    

    self.reset_state_i()

  
  def reset_state_i(self):

    self.state_i = np.zeros(shape=(1, self.num_units), dtype=np.float32)
  
  
  def locate(self, vgg16_feat):

    y_pred, self.state_i = self.sess.run([self.outputs, self.states], feed_dict={self.X_in_h: np.reshape(vgg16_feat, [1, 1, self.X_dim]), self.seq_len_h: self.len, self.init_state_h: self.state_i})
    y_pred = np.reshape(y_pred, [-1])

    return y_pred
