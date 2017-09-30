import numpy as np

class config(object):

  def __init__(self):
    super(config, self).__init__()

    self.geo_loc_path     = '/home/nvidia/turtlebot/models/loc_model/model'
    self.geo_ori_path     = '/home/nvidia/turtlebot/models/ori_model/model'
    self.vgg16_dim        = 7 * 7 * 512
    self.loc_num          = 25
    self.ori_num          = 12

    self.loc_nn_units     = 1024
    self.ori_nn_units     = 1024
    
    self.img_width        = 224
    self.img_height       = 224

    self.cam_width        = 432
    self.cam_height       = 240

    self.cam_ids          = [1, 2]
    self.cam_buffer_num   = 6

    self.img_preprocess_v = np.array([103.939, 116.779, 123.68], dtype=np.float32)