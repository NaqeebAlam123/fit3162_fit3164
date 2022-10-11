import os
import numpy as np
from facial_landmark import facial_landmark
from pathlib import Path
from constants import *
from config import *

ACTOR = config.actor

def extracting_facial_landmarks(path):
    p=Path(path)
    for path in p.iterdir():
        print('starting facial landmark extraction for ' + path.name)
        if path.name == 'neutral':
            storepath = f'{LM_ENCODER_DATASET_LANDMARK_DIR}{ACTOR}_{path.name}_1_'
        else:
            storepath = f'{LM_ENCODER_DATASET_LANDMARK_DIR}{ACTOR}_{path.name}_3_'


     
        for i in range(len(list(path.iterdir()))):
            array=facial_landmark(str(path.joinpath(str(str(i+1).rjust(3, '0'))+'.mp4')),25)
            os.makedirs(storepath+ str(str(i+1).rjust(3, '0')),exist_ok=True)
            np.save(storepath + str(str(i+1).rjust(3, '0')) + '/' + '0.npy', array)
           #print(i+1)
        print('completed for ' + path.name)


paths= f'data/video/{ACTOR}'
extracting_facial_landmarks(paths)