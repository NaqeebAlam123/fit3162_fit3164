import unittest
import audio2mfcc as a2m_convert
from unittest.mock import Mock, MagicMock, patch,call
import librosa
import io
from video_generation import draw_predicted_landmark, video_audio_compilation, video_compilation, video_generation as vd
from evaluate import evaluate
import numpy as np
import os
import torch
from Exception_classes import *
from constants import AUDIO_DATASET


class VideoGeneration(unittest.TestCase):

    def setUp(self) -> None:
        pass

    
    def test_evaluate_func(self):
        """
        purpose of testing:- 
        it is to check wether the predicted landmarks generated from the audio sample are of the required shape
        inputs:- 
        1. Two seet of audios one to depict the emoiton element and one for the content elemnt

        expected outputs:- 
        predicted landmarks are generated and is of the required shape

        actual outputs observed:-
        the method is generating the expected output as indicated by the evidence documented in the docs
        """
        # to check if errors are raise for audio files with wrong format
        with self.assertRaises(FileNotFoundError, msg="file format is incorrect"):
            evaluate('001.m4a', '002.wav')
        
        evaluate (f'{AUDIO_DATASET}/angry/001.wav', f'{AUDIO_DATASET}/contempt/001.wav')
        # check if the predicted landmark's shape matches the requirement
        predicted_lmark = torch.load('predicted_lmark.pt')
        #print(predicted_lmark.shape)
        assert predicted_lmark.shape==(95,68,2)
    
    
    def test_draw_predicted_landmarks(self):
        """
        purpose of testing:- 
        it is to check if the generated landmarks drawn on series of frames are present in the respective location
        inputs:- 
        1. store path of the frames where the predicted landmarks are drawn is to be provided

        expected outputs:- 
        frames with predicted landmarks drawn are stored in the resp location

        actual outputs observed:-
        the method is generating the expected output as indicated by the evidence documented in the docs
        """
        storepath=f'fit3162_fit3164/custom/data/image_compilation/test/'        
        # check if the predicted landmark's shape matches the requirement
        predicted_lmark = torch.load('predicted_lmark.pt')
        draw_predicted_landmark(predicted_lmark)
        
        # check existence of a single frame 
        for i in range(predicted_lmark.shape[0]):
            assert os.path.exists(storepath+'Image'+str(i)+".jpg")==True


    def test_video_compilation(self):
        """
        purpose of testing:- 
        it is to check wether the video compilation is present in its respective location

        inputs:- 
        Not applicable

        expected outputs:- 
        video compilation stored in the respective lcoation

        actual outputs observed:-
        the method is generating the expected output as indicated by the evidence documented in the docs
        """
        
        # check if the generated video is stored in the respective location
        store_video_path='fit3162_fit3164/custom/data/video_compilation'
        video_compilation()

        assert os.path.exists(os.path.join(store_video_path, 'test.mp4'))==True

    
    def test_video_audio_compilation(self):
        """
        purpose of testing:- 
        it is to check wether the video with audio compilation is present in its respective location

        inputs:- 
        content audio sample is passed as input parameter

        expected outputs:- 
        video with audio compilation stored in the respective lcoation

        actual outputs observed:-
        the method is generating the expected output as indicated by the evidence documented in the docs
        """


        video_audio_compilation(f'{AUDIO_DATASET}/angry/001.wav')
            
        # check if the video with audio file is generated or not
        assert os.path.exists(os.path.join('assets','video_test_with_audio.mp4'))==True
    
    
    
    
    
    def test_video_generation(self):
        """
        purpose of testing:- 
        it is to check wether all of the methods involved in generation of video are interacting well 
        in generation of the final result

        inputs:- 
        Not applicable

        expected outputs:- 
        video compilation stored in the respective lcoation

        actual outputs observed:-
        the method is generating the expected output as indicated by the evidence documented in the docs
        """
        # to check if errors are raised for no audio files
        with self.assertRaises(FileNotFoundError, msg="no file found"):
            vd([])

        # to check if error are raised for audio files less or more than the required 
        with self.assertRaises(OutOfBoundNumFiles, msg="file number out of bound"):
            vd(['001.wav', '002.wav', '003.wav'])
        
        
        # to check if it runs efficiently if audio files with right format are passed 
        vd([f'{AUDIO_DATASET}/angry/001.wav', f'{AUDIO_DATASET}/contempt/001.wav'])

      
        # check if the video with audio file is generated or not
        assert os.path.exists(os.path.join('assets','video_test_with_audio.mp4'))==True
      

    









if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(VideoGeneration)
    # This will run the test suite.
    unittest.TextTestRunner(verbosity=2).run(suite)
