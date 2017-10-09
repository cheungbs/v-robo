# -*- coding: utf-8 -*-

#################################################################
# todo:
# (2) label checking and correcting to check existing labels
# (3) label left, middle, right view fields separately
# (4) check and modify left, middle, right view fields labels separately

import cv2
import numpy as np
import argparse
import sys
import os

global mouseX, mouseY
global idx_frame

global frame
global frame_draw
global sel_idxs

sel_idxs = [3, 3, 3]
ui_lbls     = ['N+++', 'N++', 'N+', 'N', 'F', 'F+', 'F++', 'F+++']

d_clr = range(0, 255, 36)
d_y   = range(0, 240, 28)
d_x   = [0, 6, 12, 18, 18, 12, 6, 0]

x_l   = 0
x_m   = 166
x_r   = 333


def check_box(x, y):
  if x < x_m:
    sel_idx = 0
  elif x< x_r:
    sel_idx = 1
  else:
    sel_idx = 2

  for i in range(8):
    if y < d_y[i] + 28:
      sel_idxs[sel_idx] = i
      break

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

def draw_one_sels(frame_draw, x0, y0):
  for i in range(8):
    draw_one_item(frame_draw, i, x0, y0)

def draw_active_sels(frame_draw):
  draw_area_edge(frame_draw)

  draw_one_item(frame_draw, sel_idxs[0], x_l, 0)
  draw_sel_box(frame_draw, 0, x_l, 0)

  draw_one_item(frame_draw, sel_idxs[1], x_m, 0)
  draw_sel_box(frame_draw, 1, x_m, 0)

  draw_one_item(frame_draw, sel_idxs[2], x_r, 0)
  draw_sel_box(frame_draw, 2, x_r, 0)


def draw_ui(frame_draw, x, y):
  draw_one_sels(frame_draw, x_l, 0)
  draw_one_sels(frame_draw, x_m, 0)
  draw_one_sels(frame_draw, x_r, 0)

  check_box(x, y)
  draw_active_sels(frame_draw)

def draw_sels(event, x, y, flags, param):
  global mouseX, mouseY
  if event == cv2.EVENT_LBUTTONUP:
    frame_draw = frame.copy()

    draw_ui(frame_draw, x, y)

    cv2.imshow('frame', frame_draw)
    mouseX, mouseY = x, y

def save_labels(file_name, labels):
  np.save(file_name, labels)

def load_labels(file_name):
  labels = np.load(file_name + '.npy')
  print(file_name)

  return labels.tolist()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Labeling, Resuming, Checking mp4 lables.')
  parser.add_argument('--path', default='./tmp', help='video file directory.')
  parser.add_argument('--name', default='20170930-064841_0', help='video file name.')
  parser.add_argument('--save', action='store_true', help='save the labels.')
  parser.add_argument('--resume', action='store_true', help='in resume mode.')
  parser.add_argument('--check', action='store_true', help='in check mode.')
  args = parser.parse_args()
  print(args.save)

  cv2.namedWindow('frame')
  cv2.setMouseCallback('frame', draw_sels)

  labels   = []
  tot_lbls = 0
  if args.resume or args.check:
    lname = os.path.join(args.path, args.name)
    labels = load_labels(lname)
    tot_lbls = len(labels)
    print(tot_lbls)

  fname = os.path.join(args.path, args.name) + '.mp4'
  cap   = cv2.VideoCapture(fname)

  frames = []
  ret, frame = cap.read()
  idx_frame  = -1

  while ret:
    mouseX    = 0
    mouseY    = 0

    if args.check:
      frames.append(frame.copy())
    idx_frame += 1

    frame_draw = frame.copy()
    if idx_frame < tot_lbls:
      sel_idxs = labels[idx_frame][:]
    draw_active_sels(frame_draw)
    cv2.imshow('frame', frame_draw)

    k = cv2.waitKey(0) & 0xFF

    if idx_frame < tot_lbls:
      labels[idx_frame] = sel_idxs[:]
    else:
      sel_curr = sel_idxs[:]
      labels.append(sel_curr)
      tot_lbls += 1

    if k == 27:
      break
    elif k == ord('a'):
      print(str(idx_frame))

    ret, frame = cap.read()

  cap.release()

  tot_frms = len(frames)
  if args.check:
    while True:
      if idx_frame < 0:
        idx_frame = 0
      if idx_frame >= tot_frms:
        idx_frame = tot_frms - 1
      print(idx_frame)
      print(len(labels))
      
      sel_idxs = labels[idx_frame][:]

      frame_draw = frames[idx_frame].copy()
      draw_active_sels(frame_draw)

      cv2.imshow('frame', frame_draw)

      k = cv2.waitKey(0) & 0xFF

      labels[idx_frame] = sel_idxs[:]

      if k == 27:
        break
      if k == 81 or k == 82:
        idx_frame -= 1
      if k == 83 or k == 84:
        idx_frame += 1
        


  if args.save:
    lname = os.path.join(args.path, args.name)
    save_labels(lname, labels)
  # ll = load_labels('./tmp/20170930-064841_0')
  # print(ll)

  print('mp4 video file viewing completed.')
