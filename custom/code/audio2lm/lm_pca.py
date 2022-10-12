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
from constants import EMOTION_NET_DATASET_DIR, LM_ENCODER_DATASET_LANDMARK_DIR, LM_ENCODER_DATASET_MFCC_DIR, LANDMARK_BASICS, ACTOR

# import dlib
# import cPickle as pickle
import pickle



def lm_pca():
    """
    purpose of this function: to generate pca values 
    """
    
    mfcc_dataset = Path(os.path.join(EMOTION_NET_DATASET_DIR, 'generated_mfcc'))

    for path in mfcc_dataset.iterdir():
        #print(path.name)
        emotion, index_value = path.name.split('_')[0], path.name.split('_')[1] # path='angry_001', emotion='fear', index_value=025

        if emotion == 'neutral':
            outpath = f'{LM_ENCODER_DATASET_MFCC_DIR}{ACTOR}_neutral_1_{index_value}/'
        else:
            outpath = f'{LM_ENCODER_DATASET_MFCC_DIR}{ACTOR}_{emotion}_3_{index_value}/'

        os.makedirs(outpath, exist_ok=True)
        mfcc_all = []
        n_mfcc_files = len(list(path.iterdir()))
        for i in range(n_mfcc_files):
            mfcc = np.load(path.joinpath(str(i)+'.npy'))
            mfcc_all.append(mfcc)
        mfcc_all = np.array(mfcc_all)

        for i in range((n_mfcc_files-4)//25):
            mfcc_save = np.array(mfcc_all[i*25:i*25+25])
            np.save(f'{outpath}{str(i)}.npy', mfcc_save)

    
    train_list, val_list = [], []
    # mfcc_dataset = Path('train/landmark/dataset_{ACTOR}/mfcc/')
    mfcc_dataset = Path(LM_ENCODER_DATASET_MFCC_DIR)
    for path in mfcc_dataset.iterdir():
        for mpath in path.iterdir():
            if not Path(f'{LM_ENCODER_DATASET_LANDMARK_DIR}{path.name}/{mpath.name}').is_file():
                continue
            if int(path.name.split('_')[3]) < 10:
                val_list.append(f'{path.name}/{mpath.name}')
            else:
                train_list.append(f'{path.name}/{mpath.name}')

    
    print('dumping of training and validation set')
    print('start')
    with open(f'{LANDMARK_BASICS}train_{ACTOR}.pkl', 'wb') as f:
        pickle.dump(train_list, f)
    with open(f'{LANDMARK_BASICS}val_{ACTOR}.pkl', 'wb') as f:
        pickle.dump(val_list, f)
    print('end')

    lm_all = []
    for i in range(len(train_list)):
        path = f'{LM_ENCODER_DATASET_LANDMARK_DIR}{train_list[i]}'
        a = np.load(path)
        lm_all.append(a)

    lm = np.array(lm_all)

    lm = lm.reshape(len(train_list)*25,136)

    mean = np.mean(lm, axis=0)
    lm1 = lm - mean

    lm1 = np.array(lm1)
    U,s,V = np.linalg.svd(lm1.T)
    # print(f'U shape {U.shape} mean shape {mean.shape}')

    np.save(f'{LANDMARK_BASICS}U_68.npy', U)
    np.save(f'{LANDMARK_BASICS}mean_68.npy', mean)

if __name__ == "__main__":
    lm_pca()
