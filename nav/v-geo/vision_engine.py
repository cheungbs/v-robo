from tensorflow.contrib.keras.python.keras.applications.vgg16 import VGG16
import tensorflow as tf
import numpy as np
import cv2
import time

class vision_engine(object):
    def __init__(self, config):
        super(vision_engine, self).__init__()

        print("start vision engine loading ...")
        
        self.config      = config

        self.vgg16_graph = tf.Graph()

        with self.vgg16_graph.as_default():
            self.vgg16   = VGG16(weights='imagenet', include_top=False)

        self.init_cameras()

        print("complete vision engine loading")

    def __del__(self):
        if self.cam_num > 0:
            for cam in self.cams:
                cam.release()

    def init_cameras(self):
        self.pre_t = time.time()
        self.cams  = []
        for cam_id in self.config.cam_ids:
            cam = cv2.VideoCapture(cam_id)
            cam.set(cv2.CAP_PROP_FRAME_WIDTH , self.config.cam_width )
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.cam_height)

            if cam.isOpened():
                self.cams.append(cam)

        self.cam_num = len(self.cams)

        if self.cam_num > 0:
            self.fps            = self.cams[0].get(cv2.CAP_PROP_FPS)
            self.delta_t        = 1.0 / self.fps
            self.cam_buffer_num = self.config.cam_buffer_num
            self.delta_t_max    = self.cam_buffer_num * self.delta_t


    def sense_image(self, b_resize=False, cam_id=0):
        if self.cam_num < 1:
            print("No camera for vision.")
            return False, None

        cam = self.cams[cam_id]

        delta_t = min(time.time() - self.pre_t, self.delta_t_max)
        while delta_t >= 0:
            cam.grab()
            delta_t -= self.delta_t

        self.pre_t = time.time()

        ret, frame = cam.read()
        
        if not ret:
          print("Camera frame capture error.")
          return False, None

        if b_resize:
            frame = cv2.resize(frame, (self.config.img_height, self.config.img_width))

        return ret, frame

    def sense_images(self, b_resize=False):
        if self.cam_num < 1:
            print("No camera for vision.")
            return False, None

        frames = []
        delta_t = min(time.time() - self.pre_t, self.delta_t_max)
        while delta_t >= 0:
            for cam in self.cams:
                cam.grab()
                delta_t -= self.delta_t

        self.pre_t = time.time()

        for cam in self.cams:
            cam.grab()
        
        for cam in self.cams:
            ret, frame = cam.retrieve()
            if not ret:
                print("There are ERROR for one camera.")
                return False, None

            frames.append(frame)

        return True, frames

    def sense_vgg16(self, b_resize=False, cam_id=0):
        ret, frame = self.sense_image(b_resize, cam_id)
        if not ret:
            return False, None

        y = self.to_vgg16(frame)

        return True, y

    def sense_vgg16s(self, b_resize=False):
        ret, frames = self.sense_images(b_resize)
        if not ret:
            return False, None

        ys = []
        for frame in frames:
            y = self.to_vgg16(frame)
            ys.append(y)

        return True, ys

    def to_vgg16(self, frame):
        x  = frame.astype(np.float32)
        x -= self.config.img_preprocess_v
        x  = np.expand_dims(x, axis=0)

        with self.vgg16_graph.as_default():
            y = self.vgg16.predict(x)

        y = np.reshape(y, [self.config.vgg16_dim])

        return y

    def check_sense(self):
        if self.cam_num < 1:
            return False
        else:
            return True
