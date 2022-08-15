import time
import argparse
import os
import glob
import time
import numpy as np
import pickle 
#from torch.autograd import Variable
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
p = Path(r'D:\Project2\MFCC\MFCC\M030')
for path in p.iterdir():
# path returns the whole path to specific mfcc file D:\Project2\MFCC\MFCC\M030\surprised_029
    if path.name.split('_')[0] == 'neutral':
        storepath = r'D:\Project2\dataset\train\train\landmark\dataset_M030\mfcc\M030_'+path.name.split('_')[0]+'_1_'+path.name.split('_')[1]
    else:
        storepath = r'D:\Project2\dataset\train\train\landmark\dataset_M030\mfcc\M030_'+path.name.split('_')[0]+'_3_'+path.name.split('_')[1]
    os.makedirs(storepath,exist_ok=True)
    #print(path)
    # mfcc_all = []
    for i in range(len(list(path.iterdir()))):
        mfcc = np.load(path.joinpath(str(i)+'.npy'))
        #os.makedirs(storepath+ str(str(i+1).rjust(3, '0')),exist_ok=True)
        #print(path.joinpath(str(i)+'.npy'))
        np.save(storepath + '\\' + str(i) + '.npy', mfcc)


#get train&val lists
train_list = []
val_list = []
p = Path(r'D:\Project2\dataset\train\train\landmark\dataset_M030\mfcc')
for path in p.iterdir():
    for mpath in path.iterdir():
        #print(path.name + '\\' + mpath.name)
        #print(path.name.split('_')[-1])
        if int(path.name.split('_')[-1])<10:
            val_list.append(path.name+ '\\'+ mpath.name)
        else:
            train_list.append(path.name+ '\\' + mpath.name)

# for i in range(len(train_list)):
#     print(train_list[i])


with open(r'D:\Project2\dataset\train\train\landmark\dataset_M030\basics\train_M030.pkl', 'wb') as f:
     pickle.dump(train_list, f)
with open(r'D:\Project2\dataset\train\train\landmark\dataset_M030\basics\val_M030.pkl', 'wb') as f:
     pickle.dump(val_list, f)

#get pca(mean&U)
lm_all = []
count=0
for i in range(len(train_list)):
    path = Path(r'D:\Project2\dataset\train\train\landmark\dataset_M030\landmark')
    p=path.joinpath(train_list[i])
    #print(p)
    # check if the file exists and then add the landmark values to lm_all
    if os.path.isfile(p):
        #print('hello')
        a = np.load(p)
        lm_all.append(a)
        count+=1

lm = np.array(lm_all)
lm=lm.reshape(count*25, 136)
mean=np.mean(lm, axis=0)
lm1 = lm - mean
lm1 = np.array(lm1)
U,s,V = np.linalg.svd(lm1.T)


np.save(r'D:\Project2\dataset\train\train\landmark\dataset_M030\basics\U_68.npy', U)
np.save(r'D:\Project2\dataset\train\train\landmark\dataset_M030\basics\mean_68.npy', mean)







