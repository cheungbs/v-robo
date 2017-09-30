# _*_ coding: utf-8 -*-

import geo_loc
import geo_ori
    
class geo_engine(object):

  def __init__(self, config, geo_loc_p=None, geo_ori_p=None):
    super(geo_engine, self).__init__()

    self.config  = config

    if geo_loc_p is None:
      self.geo_loc = geo_loc.geo_loc(config)
    else:
      self.geo_loc = geo_loc_p

    if geo_ori_p is None:
      self.geo_ori = geo_ori.geo_ori(config)
    else:
      self.geo_ori = geo_ori_p

  def reset_loc(self):
    self.geo_loc.reset_state_i() 
 
  def loc(self, vgg16_feat):
    pos_v = self.geo_loc.locate(vgg16_feat)
    return pos_v

  def ori(self, vgg16_feat, loc_id):
    ori_v = self.geo_ori.ori(vgg16_feat, loc_id)
    return ori_v
