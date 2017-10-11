import tensorflow as tf
from tensorflow.contrib.keras.python.keras.applications.vgg16 import VGG16

import cv2
import numpy as np
import os
import sys
import argparse

global frame
global frame_draw
global sel_idxs

frame    = None
sel_idxs = [3, 3, 3]
ui_lbls  = ['N+++', 'N++', 'N+', 'N', 'F', 'F+', 'F++', 'F+++']

d_clr = range(0, 255, 36)
d_y   = range(0, 240, 28)
d_x   = [0, 6, 12, 18, 18, 12, 6, 0]

x_l   = 0
x_m   = 166
x_r   = 333

img_preprocess_v = np.array([103.939, 116.779, 123.68], dtype=np.float32)

def draw_area_edge(frame_draw):
  cv2.line(frame_draw, (124, 0), (124, 240), (0, 255, 0), 1)
  cv2.line(frame_draw, (308, 0), (308, 240), (0, 255, 0), 1)

def draw_one_item(frame_draw, i, x0, y0):
  clr  = d_clr[i]
  ypos = d_y[i]
  xpos = d_x[i]

  cv2.putText(frame_draw, ui_lbls[i], (x0 + 8 + xpos, y0 + 25 + ypos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (clr, 0, 255 - clr), 1, cv2.LINE_AA)

def draw_sel_box(frame_draw, sel_idx, x0, y0):
  i    = sel_idxs[sel_idx]
  clr  = d_clr[i]
  ypos = d_y[i]

  cv2.rectangle(frame_draw, (x0 + 2, y0 + 2 + ypos ), (x0 + 80, y0 + 30 + ypos), (clr, 0, 255 - clr), 0)

def draw_active_sels(frame_draw):
  draw_area_edge(frame_draw)

  draw_one_item(frame_draw, sel_idxs[0], x_l, 0)
  draw_sel_box(frame_draw, 0, x_l, 0)

  draw_one_item(frame_draw, sel_idxs[1], x_m, 0)
  draw_sel_box(frame_draw, 1, x_m, 0)

  draw_one_item(frame_draw, sel_idxs[2], x_r, 0)
  draw_sel_box(frame_draw, 2, x_r, 0)

def to_vgg16(frame, vgg16_cnn):
  x  = frame.astype(np.float32)
  x -= img_preprocess_v
  x  = np.expand_dims(x, axis=0)

  y = vgg16_cnn.predict(x)

  return y

def to_label(cls_idx):
  l, f_r = divmod(cls_idx, 64)
  f, r   = divmod(f_r    , 8)

  return [l, f, r]

def load_model(mpath):
  feat_dim  = 7 * 13 * 512
  label_dim = 512
  num_units = 1024

  feat_in_h = tf.placeholder(tf.float32, [None, feat_dim])

  logits = tf.layers.dense(feat_in_h, num_units, activation=tf.nn.relu)
  logits = tf.layers.dense(logits, num_units, activation=tf.nn.relu)
  logits = tf.layers.dense(logits, label_dim)
  outputs = tf.nn.softmax(logits)

  init = tf.global_variables_initializer()
  saver = tf.train.Saver()

  sess = tf.Session()
  sess.run(init)
  print("start loading the MLP model.")
  saver.restore(sess, mpath)
  print("the MLP model has been loaded, start to predict.")

  return sess, feat_in_h, outputs


if __name__ == '__main__':
  print('MLP Model Infer module')

  parser = argparse.ArgumentParser(description='Labeling, Resuming, Checking mp4 lables.')
  parser.add_argument('--path', default='./', help='video file directory.')
  parser.add_argument('--name', default='20170930-064841_0', help='video file name.')
  parser.add_argument('--vgg16', action='store_true', help='set to save vgg16 features')
  parser.add_argument('--show'   , action='store_true', help='show video frames when vgg16 calculation.')

  args = parser.parse_args()

  vgg16_graph = tf.Graph()
  with vgg16_graph.as_default():
    vgg16_cnn = VGG16(weights='imagenet', include_top=False)

  mpath = './models/mlp/model'
  sess, feat_in_h, outputs = load_model(mpath)

  cv2.namedWindow('frame')
  labels = []

  fname = os.path.join(args.path, args.name) + '.mp4'
  cap   = cv2.VideoCapture(fname)

  tot_frms = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  if args.vgg16:
    vgg16_feats = np.zeros([tot_lbls, 7, 13, 512], dtype=np.float32)

  for idx_frame in range(tot_frms):
    ret, frame = cap.read()

    with vgg16_graph.as_default():
      vgg16_feat = to_vgg16(frame, vgg16_cnn)

    if args.vgg16:
      vgg16_feats[idx_frame, :, :, :] = vgg16_feat

    cls_idx_v = sess.run(outputs, feed_dict={feat_in_h: vgg16_feat.reshape([1, -1])})
    cls_idx_v = cls_idx_v[0]
    cls_idx = np.argmax(cls_idx_v)

    lbl = to_label(cls_idx)
    labels.append(lbl)

    if args.show:
      frame_draw = frame.copy()
      sel_idxs = lbl[:]
      draw_active_sels(frame_draw)
      cv2.imshow('frame', frame_draw)

      cv2.waitKey(1)
    else:
      if idx_frame % 100 == 0:
        print('has proceesed ' + str(idx_frame) + ' frames.')

  sess.close()
  cap.release()

  if args.vgg16:
    vgg16_name = os.path.join(args.path, 'vgg16_feat_' + args.name)
    np.save(vgg16_name, vgg16_feats)


  print('Do you want to save the lables? (y/n)')
  while True:
    k = cv2.waitKey(0) & 0xFF
    if k == ord('y') or k == ord('Y'):
      save_name = os.path.join(args.path, 'label_' + args.name)
      np.save(save_name, labels)
      print('the labels have been saved successfully.')
      break
    elif k == ord('n') or k == ord('N'):
      print('The labels are NOT saved!!')
      break
    else:
      print('Please input y/n or Y/N')

  cv2.destroyAllWindows()

  print('The vgg16 features and labels calculation process is completed.')

  cv2.destroyAllWindows()