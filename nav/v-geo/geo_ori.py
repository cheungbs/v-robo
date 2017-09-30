from tensorflow.contrib.keras.python.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.contrib.keras.python.keras.preprocessing import image

import tensorflow as tf
import numpy as np
import cv2
class geo_ori(object):

  def __init__(self, config):
    super(geo_ori, self).__init__()

    self.config = config
    self.X_dim = self.config.vgg16_dim + self.config.loc_num
    self.y_dim = self.config.ori_num

    self.num_units = self.config.ori_nn_units

    self.graph = tf.Graph()

    with self.graph.as_default():
      self.X_in_h       = tf.placeholder(tf.float32, [None, self.X_dim])
    
      logits             = tf.layers.dense(self.X_in_h, self.num_units)
      logits             = tf.layers.dense(logits, self.y_dim)
      self.outputs       = tf.nn.softmax(logits)

      init = tf.global_variables_initializer()
      saver = tf.train.Saver()

      self.sess = tf.Session()

      self.sess.run(init)
      print("start to load geo-ori model.")
      saver.restore(self.sess, self.config.geo_ori_path)
      print("geo-ori model has been loaded.")
    
  def ori(self, vgg16_feat, loc_id):
    print('calculate orientation for location: ' + str(loc_id))
    loc_vec = np.zeros([self.config.loc_num], np.float32)
    loc_vec[loc_id] = 1.0
    loc_feat = np.concatenate([vgg16_feat.reshape([self.config.vgg16_dim]), loc_vec])
    y_pred = self.sess.run(self.outputs, feed_dict={self.X_in_h: np.reshape(loc_feat, [1, self.X_dim])})
    y_pred = np.reshape(y_pred, [-1])
    print(y_pred)

    return y_pred

  def ori_v(self, vgg16_feat, pos_v):
    loc_feat = np.concatenate([vgg16_feat.reshape([self.config.vgg16_dim]), loc_vec])
    y_pred = self.sess.run(self.outputs, feed_dict={self.X_in_h: np.reshape(loc_feat, [1, self.X_dim])})
    y_pred = np.reshape(y_pred, [-1])

    return y_pred
