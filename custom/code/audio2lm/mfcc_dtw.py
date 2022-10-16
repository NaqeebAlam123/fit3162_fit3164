# -*- coding: utf-8 -*-


import librosa
import python_speech_features
import numpy as np
import pickle
import librosa.display
import os

from dtw.dtw import dtw

from numpy.linalg import norm

import matplotlib.pyplot as plt
from config import config
from constants import AUDIO_DATASET as AUDIO_DATA, ALIGNED_AUDIO_DATA, EMOTION_LENGTH_OUTPUT

sample_interval= 0.01
window_len = 0.025
n_mfcc = 12
sample_delay =14
sample_len = 28

MEAD = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'neutral',
        'sad', 'surprised']
ACTOR = config.actor


line_1 =['001','001','001','001','001','001','001','001']
line_2 =['002','002','002','002','002','002','002','002']
line_3 =['003','003','003','003','003','003','003','003']
line_4 =['020','020','021','021','021','031','021','021']
line_5 =['021','021','022','022','022','032','022','022']
line_6 =['022','022','023','023','023','033','023','023']
line_7 =['023','023','024','024','024','034','024','024']
line_8 =['024','024','025','025','025','035','025','025']
line_9 =['025','025','026','026','026','036','026','026']
line_10=['026','026','027','027','027','037','027','027']
line_11=['027','027','028','028','028','038','028','028']
line_12=['028','029','029','029','029','039','029','029']
line_13=['029','030','030','030','030','040','030','030']
lines=[]
lines.append(line_1)
lines.append(line_2)
lines.append(line_3)
lines.append(line_4)
lines.append(line_5)
lines.append(line_6)
lines.append(line_7)
lines.append(line_8)
lines.append(line_9)
lines.append(line_10)
lines.append(line_11)
lines.append(line_12)
lines.append(line_13)

def dtw_func():
    """
    Given different emotional audio files and with same user speaking the same thing at possibly different points in time makes it difficult to encode
    content inside those audios.the goal of dynamic time warping and this function is to align two sequences of features vectors by warping the time axis repeatitively
    until optimal match is found.
    In regards to implementation of dtw itself,pairwaise comparison is done with angry emotional audio files used as reference for alignment of audios.Euclidean distance is calculated
    between two sequences is used to find the best match that minimizes overall distance.the process can be classified as something recursive as
    it needs through several paths in order to find a route that can decrease the overall distance.
    """
    # traverse every line of paths referring to similar audio files but with different emotions
    for i in range(0,13):
        m = lines[i]
        # creating path for alinged audio data
        aligned_audio = f'{ALIGNED_AUDIO_DATA}{str(i)}'
        if not os.path.exists(aligned_audio):
            os.makedirs(aligned_audio)
        for j in range(0,8):
            # creating path for audio file that needs to be extracted
            audio_path = AUDIO_DATA+MEAD[j]+'/'+m[j]+'.wav'
            # retrieve mfcc features of angry emotion audio files and save feature vectors in a file
            if(j == 0):
                # load audio files
                y1, sr1 = librosa.load(audio_path,sr=16000)
                # insert zeros at the end and the back
                y1 = np.insert(y1, 0, np.zeros(1920))
                y1 = np.append(y1, np.zeros(1920))
                # get mfcc features and store in a file
                mfcc = python_speech_features.mfcc(y1, sr1, winstep=sample_interval)
                with open(aligned_audio+'/0.pkl', 'wb') as f:
                    pickle.dump(mfcc, f)
            # compare newly genearted mfcc features for emotions other than angry with the stored one
            else:
                f = open(os.path.join(aligned_audio,'0.pkl'),'rb')
                mfcc1 = pickle.load(f)
                # load audio files
                y2, sr2 = librosa.load(audio_path, sr=16000)
                # insert zeros at the end and the back
                y2 = np.insert(y2, 0, np.zeros(1920))
                y2 = np.append(y2, np.zeros(1920))
                # get mfcc features
                mfcc2 = python_speech_features.mfcc(y2, sr2, winstep=sample_interval)
                # compare newly generated mfccs with stored one using dtw and get the best possible path
                _, _, _, path = dtw(mfcc2, mfcc1, dist=lambda x, y: norm(x - y, ord=1))
                # angry emotion mfcc features are stored in this variable
                mfcc2_n = mfcc1
                 # if we denote mfcc1 to be x and mfcc2 to be y,the path[0] will store the x coordinates and path[1] will store the y coordinates
                # Using the path coordinates ,we can assign mfcc2 values to our aligned feature vector which is mfcc2_n
                a = path[0]
                b = path[1]
                for l in range(1,len(path[0])):
                    mfcc2_n[b[l]] = mfcc2[a[l]]
                # store the mfcc features into a file
                with open(os.path.join(aligned_audio,str(j)+'.pkl'), 'wb') as f:
                    pickle.dump(mfcc2_n, f)
            # print(i,j)


    for i in range(8):
        aligned_audio = f'{EMOTION_LENGTH_OUTPUT}{str(i)}'
        if not os.path.exists(aligned_audio):
            os.makedirs(aligned_audio)
        for j in range(13):
            f = open(f'{ALIGNED_AUDIO_DATA}{str(j)}/{str(i)}.pkl','rb')
            mfcc = pickle.load(f)
            f.close()
            time_len = mfcc.shape[0]
            length = 0
            for input_idx in range(int((time_len-28)/4)+1):
                input_feat = mfcc[4*input_idx:4*input_idx+sample_len,:]
                with open(os.path.join(aligned_audio, str(length)+'.pkl'), 'wb') as f:
                    pickle.dump(input_feat, f)
                length += 1

# if __name__=="__main__":
#     dtw_func()