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


programmer = MemProgrammer(device_name = "myDAQ1",fs_acq=10000. , N = 10000,r = 5.11, states_limit= (6, 10))
programmer.setting_Ron_measurment(n_mem=1,Amp_On=1, Amp_Off=-2.5, dt_On=0.01, dt_Off=0.5,
                                    max_tests=100, max_pulses=10, saving=False)
programmer.closing()