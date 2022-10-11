import os
import pickle
import librosa
import numpy as np
import python_speech_features
from pathlib import Path
from constants import AUDIO_DATASET as AUDIO_DATA, EMOTION_NET_DATASET_DIR as MFCC_OUTPUT


def _audio2mfcc(audio_file, save):
    if not os.path.exists(save):
        os.makedirs(save)

    speech, sr = librosa.load(audio_file, sr=16000)

    # speech = np.insert(speech, 0, np.zeros(1920)) ## NOTE: 1920 zeros in front
    # speech = np.append(speech, np.zeros(1920)) ## NOTE: 1920 zeros after last position
    mfcc = python_speech_features.mfcc(speech,16000,winstep=0.01)

    time_len = mfcc.shape[0]

    for input_idx in range(int((time_len-28)/4)+1):
        input_feat = mfcc[4*input_idx:4*input_idx+28,:]
        np.save(os.path.join(save, str(input_idx)+'.npy'), input_feat)


def main():
    pathDir = os.listdir(AUDIO_DATA)
    for i in range(len(pathDir)):
        emotion = pathDir[i]
        path = os.path.join(AUDIO_DATA,emotion)
        if emotion == '.DS_Store':
            continue
        if os.path.isdir(path):
            _dir = os.listdir(path)
            for j in range(len(_dir)):
                if _dir[j] == '.DS_Store':
                    continue
                audio_file = os.path.join(path,_dir[j])
                index = _dir[j].split('.')[0]
                save = os.path.join(MFCC_OUTPUT,emotion+'_'+index)
                _audio2mfcc(audio_file, save)


    train_list, val_list = [], []
    mfcc_output_dir = Path(MFCC_OUTPUT)
    for b in mfcc_output_dir.iterdir():
        for c in b.iterdir():
            if int(b.name.split('_')[1]) < 10:
                val_list.append(b.name+'/'+c.name)
            else:
                train_list.append(b.name+'/'+c.name)

    with open('train_M030.pkl', 'wb') as f:
        pickle.dump(train_list, f)
    with open('val_M030.pkl', 'wb') as f:
        pickle.dump(val_list, f)


if __name__ == '__main__':
    main()
