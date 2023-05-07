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
from time import sleep
amps = (2, 1, -2.5)
dts = (0.05,0.05,0.05)
ranges = (10,(30,60), 100)
desired_state = 1
programmer = MemProgrammer(device_name = "myDAQ1",fs_acq=10000. , N = 10000,r = 47.5, states_limit= (100, 200))
programmer.set_3_states(desired_state, dts, amps, ranges)
programmer.closing()
