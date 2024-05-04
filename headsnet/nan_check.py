# https://github.com/mrbid
import sys
import numpy as np
from time import time_ns

# print everything / no truncations
np.set_printoptions(threshold=sys.maxsize)

# load training data
st = time_ns()
print("Loading...")
data = []
with open("train_x.dat", 'rb') as f: data = np.fromfile(f, dtype=np.float32)
print("[train_x] NaN's detected:", np.count_nonzero(np.isnan(data)))
data = []
with open("train_y.dat", 'rb') as f: data = np.fromfile(f, dtype=np.float32)
print("[train_y] NaN's detected:", np.count_nonzero(np.isnan(data)))
timetaken = (time_ns()-st)/1e+9
print("Time Taken:", "{:.2f}".format(timetaken), "seconds")