import os
import numpy as np
from facial_landmark import facial_landmark
from pathlib import Path
from constants import *
from config import *
from Exception_classes import * 

def extracting_facial_landmarks(path, test_identifier=False):
    p=Path(path)
    # If path to directory does not exist, raise a PathNotFoundError
    if  not os.path.exists(p):
        raise PathNotFoundError(path,"Incorrect Directory Path")
    else:
        # check the path
        for paths in p.iterdir():
            # if path to the vdeo doesn't exist, raise a PathNotFoundError
            if  not os.path.exists(paths):
                raise PathNotFoundError(path,"Incorrect Directory Path")

    p=Path(path)        
    for path in p.iterdir():
    
        print('starting facial landmark extraction for ' + path.name)
        if path.name == 'neutral':
            storepath = f'{LM_ENCODER_DATASET_LANDMARK_DIR}{ACTOR}_{path.name}_1_'
        else:
            storepath = f'{LM_ENCODER_DATASET_LANDMARK_DIR}{ACTOR}_{path.name}_3_'
        

        
        for i in range(len(list(path.iterdir()))):
            if test_identifier==True:
                array=facial_landmark(str(path.joinpath(str(str(i+1).rjust(3, '0'))+'.mp4')),25)
                os.makedirs(storepath+ str(str(i+1).rjust(3, '0')),exist_ok=True)
                np.save(storepath + str(str(i+1).rjust(3, '0')) + '/' + '0.npy', array)
                break
           #print(i+1)
        print('completed for ' + path.name)
        
        # check only for one sample in testing
        if test_identifier==True:
            break

# paths= f'data/video/{ACTOR}'
# extracting_facial_landmarks(paths)

