import os
import pickle
import librosa
import numpy as np
import python_speech_features
from pathlib import Path
from Exceptions_Classes import PathNotFoundError

AUDIO_DATA = 'data/audio/M030_wav'
MFCC_OUTPUT = 'data/mfcc/M030/generated_mfcc'


# AUDIO_DATA = 'custom/data/audio/M030_wav'
# MFCC_OUTPUT = 'custom/data/mfcc/M030/generated_mfcc'

def _audio2mfcc(audio_file, save,test_identifier=False):
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

def main(audio_file,mfcc_path,test_identifier=False):
    # roots = 'data/audio/M030_wav\contempt'
    # files ['001.wav', '002.wav', ....]

    iterator=os.walk(audio_file,topdown=True)
    first=next(iterator)
    Continue_iteration=True
    emotion_names=first[1]
    AUDIO_DATA=os.path.abspath(audio_file)
    if not(len(emotion_names)>0):
        raise PathNotFoundError(audio_file, "Path does not lead to folder or directory", 5)
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
                 save_path = os.path.join(mfcc_path, emotion_names[emotion_num] + '_' + index)
                 _audio2mfcc(audio_file_path, save_path,test_identifier)
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

    os.makedirs('data/mfcc/M030/basics')
    with open('data/mfcc/M030/basics/train_M030.pkl', 'wb') as f:
        pickle.dump(train_list, f)
    with open('data/mfcc/M030/basics/val_M030.pkl', 'wb') as f:
        pickle.dump(val_list, f)

   
    # print(train_list==train_list_x)
    # print(val_list==val_list_x)


if __name__ == '__main__':
    AUDIO_DATA = 'data/audio/M030_wav'
    MFCC_OUTPUT = 'data/mfcc/M030/generated_mfcc'
    # with open('custom/data/mfcc/M030/basics/val_M030.pkl', 'rb') as f:
    #     data=pickle.load(f)
    # for l in data:
    #     x=l.split('/')[0]
    #     if int(x.split('_')[1])>10:
    #         print('error')
    # print(data)
    try:
     main(AUDIO_DATA,MFCC_OUTPUT,False)
    except PathNotFoundError as e:
        print(e)








