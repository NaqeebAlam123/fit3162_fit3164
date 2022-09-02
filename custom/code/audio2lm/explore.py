import os
import numpy as np

curr_path = os.getcwd() + '/data/landmark/M030/'

for file in os.listdir(curr_path):
    np_file = os.listdir(os.path.join(curr_path, file))[0]
    _np = np.load(os.path.join(curr_path, file, np_file))
    print(_np.shape)
