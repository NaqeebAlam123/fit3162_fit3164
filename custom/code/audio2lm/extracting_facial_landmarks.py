import os
import numpy as np
from facial_landmarks import facial_landmark
from pathlib import Path

def extracting_facial_landmarks(path):
    """
    This function uses the facial landmark function in a separate file to generate 25 frames landmark points
    for each video in the desired path

    Parameters
    ----------
    path: path that holds collection of videos

    """

    # create posix path and find directories starting with the keyword (path) along with the associated sub-directories
    p=Path(path)
    # iterate through all directories
    for path in p.iterdir():

        print('starting facial landmark extraction for ' + path.name)
        # provide the file name with index 1 if it is a neutral emotional audio file otherwise assign it 3
        if path.name == 'neutral':
            storepath = "../../../dataset/train/train/landmark/dataset_M030/landmark/M030_"+path.name+'_1_'
        else:
            storepath = "../../../dataset/train/train/landmark/dataset_M030/landmark/M030_"+path.name+'_3_'

        # iterate through all video files in a single emotional directory
        for i in range(len(list(path.iterdir()))):
            array=facial_landmark(str(path.joinpath(str(str(i+1).rjust(3, '0'))+'.mp4')),25)
            os.makedirs(storepath+ str(str(i+1).rjust(3, '0')),exist_ok=True)
            np.save(storepath + str(str(i+1).rjust(3, '0')) + '/' + '0.npy', array)
        print('completed for ' + path.name)





if __name__ == '__main__':
    path= "../../../dataset/train/train/landmark/dataset_M030/video"
    extracting_facial_landmarks(path)