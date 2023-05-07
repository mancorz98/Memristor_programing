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
amps = [1.5]
dts = [0.05]
for amp in amps:
    for dt in dts:
        programmer = MemProgrammer(device_name = "myDAQ1",fs_acq=10000. , N = 10000,r = 47.5, states_limit= (100, 200))
        programmer.setting_Ron_measurment(n_mem=4,Amp_On=amp, Amp_Off=-2.5, dt_On=dt, dt_Off=0.1,
                                        max_tests=20, max_pulses=10, saving=True)
        programmer.closing()
        sleep(5*60)
        