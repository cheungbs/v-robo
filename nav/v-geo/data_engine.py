# -*- coding: utf-8 -*-

import time
import cv2
import rospy

import actuator_engine
import vision_engine

class data_engine(object):
    def __init__(self, config, vision_engine_p=None, actuator_engine_p=None):
        super(data_engine, self).__init__()

        print("start data engine loading ...")

        self.save_path = "/home/nvidia/VQA/data/tmp/"

        if vision_engine_p is None:
            self.vision_engine = vision_engine.vision_engine(config)
        else:
            self.vision_engine = vision_engine_p

        if actuator_engine_p is None:
            self.actuator_engine = actuator_engine.actuator_engine(config)
        else:
            self.actuator_engine = actuator_engine_p

        self.fourcc     = cv2.VideoWriter_fourcc(*'MP4V')
        self.vid_suffix = '.mp4'
        self.vid_fps    = 10
        self.vid_files  = None
        self.vid_rate = rospy.Rate(self.vid_fps)

        self.eposode_id   = None
        self.eposode_file = None

        print("complete data engine loading")

    def __del__(self):
        if self.eposode_file is not None:
            self.eposode_file.close()
            self.eposode_file = None

        if self.vid_files is not None:
            for cam_id in range(self.vision_engine.cam_num):
                self.vid_files[cam_id].release()

        self.vid_files  = None
        self.eposode_id = None

    def eposode_start(self):
        self.eposode_id   = time.strftime("%Y%m%d-%H%M%S")
        self.eposode_file = open(self.save_path + 'eposode_' + self.eposode_id + '.txt', 'a')
        self.eposode_file.write(self.eposode_id + '\n')
        self.eposode_file.write(str(self.vision_engine.cam_num) + '\n')
        self.eposode_file.flush()

    def eposode_end(self):
        if self.eposode_file is not None:
            self.eposode_file.close()
            self.eposode_file = None
        self.eposode_id   = None

    def eposode_start_vid(self):
        self.eposode_id   = time.strftime("%Y%m%d-%H%M%S")
        self.eposode_file = open(self.save_path + 'eposode_' + self.eposode_id + '.txt', 'a')
        self.eposode_file.write(self.eposode_id + '\n')
        self.eposode_file.write(str(self.vision_engine.cam_num) + '\n')

        self.vid_files = []
        for cam_id in range(self.vision_engine.cam_num):
            vid_name = self.save_path + self.eposode_id + '_' + str(cam_id) + self.vid_suffix
            vout = cv2.VideoWriter(vid_name, self.fourcc, self.vid_fps, (self.vision_engine.config.cam_width, self.vision_engine.config.cam_height))
            self.vid_files.append(vout)

    def eposode_end_vid(self):
        if self.eposode_file is not None:
            self.eposode_file.close()
            self.eposode_file = None

        if self.vid_files is not None:
            for cam_id in range(self.vision_engine.cam_num):
                self.vid_files[cam_id].release()

        self.vid_files    = None
        self.eposode_id   = None

    def record_one_frame(self):
        ret, cam_imgs = self.vision_engine.sense_images()
        if not ret:
            return False

        for cam_id in range(self.vision_engine.cam_num):
            self.vid_files[cam_id].write(cam_imgs[cam_id])

        return True


    def location_data(self):
        cache = []
        loop_rate, N_loop = self.actuator_engine.get_turn_loop_rate('normal')

        for _ in range(10):
            self.vision_engine.sense_images()
            self.actuator_engine.a_turn_left()
            loop_rate.sleep()

        for _ in range(N_loop):
            self.capture_images_cache(cache)
            self.actuator_engine.a_turn_left()
            loop_rate.sleep()

        return self.add_to_eposode(cache, 'loc', 'left')

    def forward_data(self, dist='0'):
        if dist is None:
            dist = '0'

        cache = []
        loop_rate, N_loop = self.actuator_engine.get_forward_loop_rate(dist)

        for i in range(N_loop):
            self.capture_images_cache(cache)
            self.actuator_engine.a_forward()
            loop_rate.sleep()

        return self.add_to_eposode(cache, dist, 'forward')

    def turn_data(self, angle='0'):
        if angle is None:
            angle = '0'

        cache = []
        loop_rate, action, N_loop, ori = self.actuator_engine.get_ori_turn_loop_rate(angle)
        if ori == 0:
            lbl = 'left'
        else:
            lbl = 'right'

        for _ in range(N_loop):
            self.capture_images_cache(cache)
            action()
            loop_rate.sleep()

        return self.add_to_eposode(cache, angle, lbl)

    def a_action_data(self, a_action_name):
        ret, cam_imgs = self.vision_engine.sense_images()
        if not ret:
            return False

        sub_eposode_id = time.strftime("%Y%m%d-%H%M%S")
        self.eposode_file.write(sub_eposode_id + ' 1 ' + a_action_name + ' ' + a_action_name + '\n')
        self.eposode_file.flush()

        fname_pre = self.eposode_id + '_' + sub_eposode_id
        for idx_cam in range(self.vision_engine.cam_num):
            cam_img = cam_imgs[idx_cam]
            fname = self.save_path + fname_pre + '_0_' + str(idx_cam) + '.jpg'

            cv2.imwrite(fname, cam_img)

        return True


    def capture_image(self):
        ret, frame = self.vision_engine.sense_image()
        if not ret:
            return False
        fname = "img_" + time.strftime("%Y%m%d-%H%M%S") + ".jpg"
        cv2.imwrite(self.save_path + fname, frame)

        return True

    def capture_images(self):
        ret, frames = self.vision_engine.sense_images()
        if not ret:
            return False

        fname_pre = "img_" + time.strftime("%Y%m%d-%H%M%S")
        for i in range(len(frames)):
            fname = str(i) + '/' + fname_pre + "_" + str(i) + ".jpg"
            cv2.imwrite(self.save_path + fname, frames[i])

        return True

    def capture_images_cache(self, cache):
        ret, frames = self.vision_engine.sense_images()
        if not ret:
            return False

        cache.append(frames)

        return True

    def add_to_eposode(self, cache, sub_eposode_name, a_action_name):
        if self.eposode_id is None or self.eposode_file is None:
            print('ERROR for eposode ID and FILE.')
            return False

        frames_num = len(cache)
        sub_eposode_id = time.strftime("%Y%m%d-%H%M%S")
        self.eposode_file.write(sub_eposode_id + ' ' + str(frames_num) + ' ' + sub_eposode_name + ' ' + a_action_name + '\n')
        self.eposode_file.flush()

        fname_pre = self.eposode_id + '_' + sub_eposode_id
        for idx_frame in range(frames_num):
            cam_imgs = cache[idx_frame]
            for idx_cam in range(self.vision_engine.cam_num):
                cam_img = cam_imgs[idx_cam]
                fname = self.save_path + fname_pre + '_' + str(idx_frame) + '_' + str(idx_cam) + '.jpg'

                cv2.imwrite(fname, cam_img)

        return True

    def excute_cmds(self, cmd_str, cmd_param):
        if cmd_str == 'g':
            self.forward_data(cmd_param)
        elif cmd_str in ['gf', 'gff', 'gfff', 'gn', 'gnn', 'gnnn']:
            self.forward_data(cmd_str)
        elif cmd_str == 't':
            self.turn_data(cmd_param)
        elif cmd_str in ['fl', 'fr', 'lf', 'rf', 'l', 'r', 'lb', 'rb', 'bl', 'br', 'b']:
            self.turn_data(cmd_str)
        elif cmd_str in ['X-Button', 'Arrow-Button-Left']:
            self.a_action_data('left')
            self.actuator_engine.a_turn_left()
        elif cmd_str in ['B-Button', 'Arrow-Button-Right']:
            self.a_action_data('right')
            self.actuator_engine.a_turn_right()
        elif cmd_str in ['LT-Button', 'RT-Button']:
            self.a_action_data('forward')
            self.actuator_engine.a_forward()
        elif cmd_str == 'loc':
            self.location_data()
        elif cmd_str == 'cap':
            self.capture_image()
        elif cmd_str == 'caps':
            self.capture_images()
        