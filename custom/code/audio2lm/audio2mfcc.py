import os
import pickle
import librosa
import numpy as np
import python_speech_features
from pathlib import Path
from Exception_classes import *
from constants import *

# AUDIO_DATA = 'data/audio/M030_wav'
# MFCC_OUTPUT = 'data/mfcc/M030'

def audio_to_mfcc_representation(audio_file, save,test_identifier=False):
    if os.path.exists(audio_file):
      speech, sr = librosa.load(audio_file, sr=16000)
    else:
      raise PathNotFoundError(audio_file,"Either the file is not found or path is invalid",1)

    # speech = np.insert(speech, 0, np.zeros(1920)) ## NOTE: 1920 zeros in front
    # speech = np.append(speech, np.zeros(1920)) ## NOTE: 1920 zeros after last position
    
    mfcc = python_speech_features.mfcc(speech,16000,winstep=0.01)
    time_len = mfcc.shape[0]
    if test_identifier:
     print(time_len)
    if not os.path.exists(save):
        os.makedirs(save)
    for input_idx in range(int((time_len-28)/4)+1):
        feat_input = mfcc[4*input_idx:4*input_idx+28,:]
        np.save(os.path.join(save, str(input_idx)+'.npy'), feat_input)



def main(AUDIO_DATA, MFCC_OUTPUT, test_identifier=False):
     # roots = 'data/audio/M030_wav\contempt'
    # files ['001.wav', '002.wav', ....]

    iterator=os.walk(AUDIO_DATA,topdown=True)
    first=next(iterator)
    Continue_iteration=True
    emotion_names=first[1]
    AUDIO_DATA=os.path.abspath(AUDIO_DATA)
    if not(len(emotion_names)>0):
        raise PathNotFoundError(AUDIO_DATA, "Path does not lead to folder or directory", 5)
    emotion_num=0
    while Continue_iteration:
        try:
             tuple=next(iterator)
             root=tuple[0]
             files=tuple[2]
           
             for file in files:
                 audio_file_path=os.path.join(root,file)
                 ext = file.split(".")
                 if (len(ext) > 1):
                     ext = ext[1]
                 if ext!="wav":
                    raise PathNotFoundError(audio_file_path,"Path is leading into an incorrect file",4)
                 index = file.split('.')[0]
                 save_path = os.path.join(MFCC_OUTPUT, emotion_names[emotion_num] + '_' + index)
                 audio_to_mfcc_representation(audio_file_path, save_path,test_identifier)
             emotion_num=emotion_num+1
        except StopIteration:
            Continue_iteration=False

    print('start')
    print('segregation of validation and training set')
    train_list, val_list = [], []
    index = 0
    for (roots, dir, files) in os.walk(MFCC_OUTPUT):
        if index > 0:
            for file in files:
                if int(sub_folders[index - 1].split('_')[1]) < 10:
                    val_list.append(sub_folders[index - 1] + '/' + file)
                else:
                    train_list.append(sub_folders[index - 1] + '/' + file)
        else:
            sub_folders = dir
        index += 1
    print('end')

    print('dumping of training and validation set')
    print('start')
    if not os.path.exists(f'{EMOTION_NET_DATASET_DIR}basics'):
        os.makedirs(f'{EMOTION_NET_DATASET_DIR}basics')
    with open(f'{EMOTION_NET_DATASET_DIR}basics/train_{ACTOR}.pkl', 'wb') as f:
        pickle.dump(train_list, f)
    with open(f'{EMOTION_NET_DATASET_DIR}basics/val_{ACTOR}.pkl', 'wb') as f:
        pickle.dump(val_list, f)
    print('end')
   