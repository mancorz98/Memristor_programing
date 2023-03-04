import nidaqmx
from nidaqmx import constants, stream_readers, stream_writers
import scipy.signal as signal

import time
import numpy as np
import matplotlib.pyplot as plt
import string as str
from datetime import datetime
import os
from MemProgrammer import *


programmer = MemProgrammer(device_name = "myDAQ1",fs_acq=10000. , N = 10000,r = 4.7, states_limit= (4, 50))
#programmer.setting_Ron_once()
R = programmer.check_resistane()
print(R)
#print(state)
programmer.closing()