import time
import actuator_engine
import vision_engine

min_delay = 1.0 / 30.0
speed_delay_dict = {
	'fastest' : min_delay,
	'faster'  : min_delay * 2,
	'fast'    : min_delay * 4,
	'normal'  : min_delay * 8,
	'slow'    : min_delay * 16,
	'slower'  : min_delay * 32,
	'slowest' : min_delay * 64
}

class excute_engine(object):
	"""excute_engine class"""
	def __init__(self, config, vision_engine_p=None, actuator_engine_p=None):
		super(excute_engine, self).__init__()
		self.config = config

		if actuator_engine_p is None:
			self.actuator_engine = actuator_engine.actuator_engine(config)
		else:
			self.actuator_engine = actuator_engine_p

		if vision_engine_p is None:
			self.vision_engine = vision_engine.vision_engine(config)
		else:
			self.vision_engine = vision_engine_p


	def exec_macro_behavior(self, action):
		pass

	# atomic action
	# action = {
	#     'vision': None/1/2
	#     'motion': None/left/right/forward
	#     'speed' : None/fastest/faster/fast/normal/slow/slower/slowest --> None === normal
	#     'record': None/True/False    ===> True for training
	# }
	def exec_atom_behavior(self, action):
		pass