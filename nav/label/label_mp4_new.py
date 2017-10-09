# -*- coding: utf-8 -*-

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

frame    = None
sel_idxs = [3, 3, 3]
ui_lbls  = ['N+++', 'N++', 'N+', 'N', 'F', 'F+', 'F++', 'F+++']

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
  global frame

  if event == cv2.EVENT_LBUTTONUP:
    if frame is None:
      return

    frame_draw = frame.copy()

    draw_ui(frame_draw, x, y)

    cv2.imshow('frame', frame_draw)
    mouseX, mouseY = x, y

def save_labels(file_name, labels):
  np.save(file_name, labels)

def load_labels(file_name):
  labels = np.load(file_name + '.npy')

  return labels.tolist()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Labeling, Resuming, Checking mp4 lables.')
  parser.add_argument('--path', default='./', help='video file directory.')
  parser.add_argument('--name', default='20170930-064841_0', help='video file name.')
  parser.add_argument('--iframe', default=0, type=int, help='nav to the iframe-th video frame.' )
  parser.add_argument('--save', action='store_true', help='save the labels.')
  parser.add_argument('--resume', action='store_true', help='in resume mode.')
  parser.add_argument('--check', action='store_true', help='in check mode.')
  parser.add_argument('--nav', action='store_true', help='nav for labeling.')
  args = parser.parse_args()

  cv2.namedWindow('frame')
  cv2.setMouseCallback('frame', draw_sels)

  labels   = []
  tot_lbls = 0
  if args.nav or args.resume or args.check:
    lname = os.path.join(args.path, args.name)
    labels = load_labels(lname)
    tot_lbls = len(labels)
    print('Labels has been loaded, ' + str(tot_lbls) + ' frames have been labeled.')

  fname = os.path.join(args.path, args.name) + '.mp4'
  cap   = cv2.VideoCapture(fname)

  tot_frms = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  frames = []

  if not args.nav:
    for idx_frame in range(tot_frms):

      ret, frame = cap.read()

      mouseX    = 0
      mouseY    = 0

      if args.check:
        frames.append(frame.copy())

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
        print('current frame index is: ' + str(idx_frame))

      ret, frame = cap.read()
  else:
    print('Waiting for loading video frames ... ')
    for _ in range(tot_frms):
      ret, frame = cap.read()
      frames.append(frame.copy())

    if tot_lbls < tot_frms:
      for _ in range(tot_frms - tot_lbls):
        labels.append([3, 3, 3])

    tot_lbls = len(labels)
    idx_frame = args.iframe
    print('Video frames are all loaded. total ' + str(tot_frms) + ' frames.')
    print('Locate to the frame ' + str(idx_frame))

  cap.release()

  tot_frms = len(frames)
  if args.check or args.nav:
    print('Totally ' + str(tot_frms) + ' Video frames are loaded, use ARROW buttons to navigate videos for labelling.')

    while True:
      mouseX    = 0
      mouseY    = 0

      if idx_frame < 0:
        idx_frame = 0
      if idx_frame >= tot_frms:
        idx_frame = tot_frms - 1
      
      sel_idxs = labels[idx_frame][:]

      frame = frames[idx_frame]
      frame_draw = frame.copy()
      draw_active_sels(frame_draw)

      cv2.imshow('frame', frame_draw)

      k = cv2.waitKey(0) & 0xFF

      labels[idx_frame] = sel_idxs[:]

      if k == 27:
        break
      if k == ord('s') or k == ord('k'):
        if idx_frame < 1:
          print('current frame is the first frame, no previous frame.')
        else:
          idx_frame -= 1
      if k == ord('d') or k == ord('l'):
        if idx_frame >= tot_frms - 1:
          print('current frame is the last frame, no next frame.')
        idx_frame += 1
      if k == ord('a'):
        print('current frame index: ' + str(idx_frame))


  if args.save:
    lname = os.path.join(args.path, args.name)
    save_labels(lname, labels)
  else:
    print('Do you want to save the lables? (y/n)')

    while True:
      k = cv2.waitKey(0) & 0xFF
      if k == ord('y') or k == ord('Y'):
        lname = os.path.join(args.path, args.name)
        save_labels(lname, labels)
        print('the labels have been saved successfully.')
        break
      elif k == ord('n') or k == ord('N'):
        print('The labels are NOT saved!!')
        break
      else:
        print('Please input y/n or Y/N')

  cap.release()
  print('The labelling process is terminated.')

