#!/usr/bin/env python2
# -*- coding: utf-8 -*-

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
# import utils
from tqdm import tqdm
import torchvision.transforms as transforms
import shutil
from collections import OrderedDict
import python_speech_features
from skimage import transform as tf
from scipy.spatial import procrustes
from mfcc2lm import EMOTION_NET_DATASET_DIR, LM_ENCODER_DATASET_LANDMARK_DIR, LM_ENCODER_DATASET_MFCC_DIR

# import dlib
# import cPickle as pickle
import pickle

def default_parameter_reader(flist):
    parameter_list = []
    name_list = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            name = line.split(' ')[-1]
            parameters = line.split(' ')[:-1]
            for i in range(len(parameters)):
                parameters[i] = float(parameters[i])
            parameter_list.append(parameters)
            name_list.append(name)
    return name_list, parameter_list


def parameter_reader(flist):
    parameter_list = []
 #   name_list = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
          #  name = line.split(' ')[-1]
            parameters = line.split(' ')
            for i in range(len(parameters)):
                parameters[i] = float(parameters[i])
            parameter_list.append(parameters)
        #    name_list.append(name)
    return parameter_list

# def trainFile(filepath):
#     t=0
#     pathDir =  os.listdir(filepath)
#     for j in range(len(pathDir)): #len(pathDir)
#         allDir = pathDir[j]
#         intensity = int(allDir.split('_')[2])
#         index = int(allDir.split('_')[1])
#         if (index >61):
#             path_68 = '/home/thea/data/MEAD/ATnet_emotion/dataset/68_landmark/'+allDir
#             path_106 = '/home/thea/data/MEAD/ATnet_emotion/dataset/106_landmark/'+allDir
#             path_mfcc = '/home/thea/data/MEAD/ATnet_emotion/dataset/MFCC/'+allDir
#             if not os.path.exists(path_68):
#                 os.makedirs(path_68)
#             if not os.path.exists(path_106):
#                 os.makedirs(path_106)
#             if not os.path.exists(path_mfcc):
#                 os.makedirs(path_mfcc)
#             path = os.path.join(filepath,allDir)
#             lm_path = os.path.join(path,'landmark.txt')
#             cp_path = os.path.join(path,'crop_lmk.txt')
#             name,parameters = default_parameter_reader(cp_path)
#             para =  parameter_reader(lm_path)

#             mfcc_path = '/home/thea/data/MEAD/ATnet_emotion/dataset/mfcc/'+allDir+'.npy'
#             mfcc = np.load(mfcc_path)
#             time_len = mfcc.shape[0]
#             sample_len = 28
#             mfcc_all = []
#             for input_idx in range(int((time_len-28)/4)+1):
#                 input_feat = mfcc[4*input_idx:4*input_idx+sample_len,:]
#                 mfcc_all.append(input_feat)

#             if(len(parameters)==len(para)):
#                 n = (len(para)-4)//25
#                 for i in range(n):
#                     para_68 = np.array(para[i*25:i*25+25])
#                     np.save('/home/thea/data/MEAD/ATnet_emotion/dataset/68_landmark/'+allDir+'/'+str(i)+'.npy', para_68)
#                     para_106 = np.array(parameters[i*25:i*25+25])
#                     para_106 = para_106[:,6:]/255
#                     np.save(path_106+'/'+str(i)+'.npy', para_106)
#                     mfcc_save = np.array(mfcc_all[i*25:i*25+25])
#                     np.save(path_mfcc + '/'+str(i)+'.npy',mfcc_save)
#             print(allDir, n)

#mfcc file split
# p = Path('train/MFCC/M030/')
mfcc_dataset = Path(EMOTION_NET_DATASET_DIR)

for path in mfcc_dataset.iterdir():
    _emotion, idx = path.name.split('_')[0], path.name.split('_')[1]
    if _emotion == 'neutral':
        outpath = f'{LM_ENCODER_DATASET_MFCC_DIR}/M030_neutral_1_{idx}'
    else:
        outpath = f'{LM_ENCODER_DATASET_MFCC_DIR}/M030_{_emotion}_3_{idx}'

    os.makedirs(outpath, exist_ok=True)
    mfcc_all = []
    for i in range(len(list(path.iterdir()))):
        mfcc = np.load(path.joinpath(str(i)+'.npy'))
        mfcc_all.append(mfcc)
    mfcc_all = np.array(mfcc_all)
    n = (len(list(path.iterdir()))-4)//25
    for j in range(n):
        mfcc_save = np.array(mfcc_all[j*25:j*25+25])
        np.save(outpath + '/'+str(j)+'.npy', mfcc_save)


#get train&val lists
train_list, val_list = [], []

# mfcc_dataset = Path('train/landmark/dataset_M030/mfcc/')
mfcc_dataset = Path(LM_ENCODER_DATASET_LANDMARK_DIR)
for path in mfcc_dataset.iterdir():
    for mpath in path.iterdir():
        if int(path.name.split('_')[3]) < 10:
            val_list.append(path.name+'/'+mpath.name)
        else:
            train_list.append(path.name+'/'+mpath.name)

with open('code/audio2lm/landmark/basics/train_M030.pkl', 'wb') as f:
    pickle.dump(train_list, f)
with open('code/audio2lm/landmark/basics/val_M030.pkl', 'wb') as f:
    pickle.dump(val_list, f)

#get pca(mean&U)
lm_all = []
for i in range(len(train_list)):
    # path = 'landmark/train/landmark/dataset_M030/landmark/'+train_list[i]
    path = f'{LM_ENCODER_DATASET_LANDMARK_DIR}{train_list[i]}'
    a = np.load(path)
    lm_all.append(a)
lm = np.array(lm_all)
# lm = lm.reshape(len(train_list)*25,212)
mean = np.mean(lm, axis=0)

lm1 = lm - mean

lm1 = np.array(lm1)
U,s,V = np.linalg.svd(lm1.T)

np.save('code/audio2lm/landmark/basics/U_68.npy', U)
np.save('code/audio2lm/landmark/basics/mean_68.npy', mean)
