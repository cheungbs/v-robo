# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse
import sys
import os

from tensorflow.contrib.keras.python.keras.applications.vgg16 import VGG16
import tensorflow as tf

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

def load_labels(file_name):
  labels = np.load(file_name + '.npy')
  return labels.tolist()

def save_vgg16_labels(file_name, vgg16_labels):
  np.save(file_name, vgg16_labels)

def load_vgg16_labels(file_name):
  vgg16_labels = np.load(file_name + '.npy')
  return vgg16_labels


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Labeling, Resuming, Checking mp4 lables.')
  parser.add_argument('--path', default='./', help='video file directory.')
  parser.add_argument('--name', default='20170930-064841_0', help='video file name.')
  parser.add_argument('--show'   , action='store_true', help='show video frames when vgg16 calculation.')

  args = parser.parse_args()

  vgg16_cnn = VGG16(weights='imagenet', include_top=False)

  cv2.namedWindow('frame')

  vgg16_cls = []

  lname = os.path.join(args.path, 'label_' + args.name)
  labels = load_labels(lname)
  tot_lbls = len(labels)
  print('Labels has been loaded, ' + str(tot_lbls) + ' frames have been labeled.')

  vgg16_feats = np.zeros([tot_lbls, 7, 13, 512], dtype=np.float32)

  fname = os.path.join(args.path, args.name) + '.mp4'
  cap   = cv2.VideoCapture(fname)

  tot_frms = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  assert(tot_lbls <= tot_frms)

  for idx_frame in range(tot_lbls):

    ret, frame = cap.read()

    vgg16_feat = to_vgg16(frame, vgg16_cnn)
    vgg16_feats[idx_frame, :, :, :] = vgg16_feat

    lbl     = labels[idx_frame]
    lbl_cls = lbl[0] * 64 + lbl[1] * 8 + lbl[2]
    vgg16_cls.append(lbl_cls)

    if args.show:
      frame_draw = frame.copy()
      sel_idxs = labels[idx_frame][:]
      draw_active_sels(frame_draw)
      cv2.imshow('frame', frame_draw)

      cv2.waitKey(1)
    else:
      if idx_frame % 100 == 0:
        print('has proceesed ' + str(idx_frame) + ' frames.')

  cap.release()

  vgg16_feats_labels = {'feats': vgg16_feats, 'labels': vgg16_cls}

  print('Do you want to save the vgg16 features and lables? (y/n)')
  while True:
    k = cv2.waitKey(0) & 0xFF
    if k == ord('y') or k == ord('Y'):
      save_name = os.path.join(args.path, 'vgg16_label_' + args.name)
      save_vgg16_labels(save_name, vgg16_feats_labels)
      print('the vgg16 features and labels have been saved successfully.')
      break
    elif k == ord('n') or k == ord('N'):
      print('The vgg16 features and labels are NOT saved!!')
      break
    else:
      print('Please input y/n or Y/N')

  cv2.destroyAllWindows()

  print('The vgg16 features and labels calculation process is completed.')
