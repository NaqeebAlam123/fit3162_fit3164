import time
import argparse
import os
import glob
import time
import numpy as np

from torch.autograd import Variable
import librosa
from pathlib import Path
import re

import cv2
import scipy.misc
#import utils
from tqdm import tqdm
import torchvision.transforms as transforms
import shutil
from collections import OrderedDict
import python_speech_features
from skimage import transform as tf
from copy import deepcopy
from scipy.spatial import procrustes

import dlib
#import cPickle as pickle

#mfcc file split
p = Path(r'D:\Project1\fit3162_fit3164\train\MFCC\M030')
for path in p.iterdir():
# path returns the whole path to specific mfcc file D:\Project1\fit3162_fit3164\train\MFCC\M030\surprised_029
    if path.name.split('_')[0] == 'neutral':
        outpath = r'D:\Project1\fit3162_fit3164\dataset\train\train\landmark\dataset_M030\mfcc\M030_'+path.name.split('_')[0]+'_1_'+path.name.split('_')[1]
    else:
        outpath = r'D:\Project1\fit3162_fit3164\dataset\train\train\landmark\dataset_M030\mfcc\M030_'+path.name.split('_')[0]+'_3_'+path.name.split('_')[1]






