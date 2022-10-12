import unittest
import facial_landmark as fl
from Exception_classes import *
import dlib
from unittest.mock import patch
import extracting_facial_landmarks as f2
import lm_pca as lm
import numpy as np
from constants import * 
import os

class FacialLandmarks(unittest.TestCase) :

    def setUp(self) -> None:
        pass

    def test_extracting_frames(self):

        # to check the extracting_frames's methods behaviour if a wrong video path is provided
        video_path="/readme.mp4"
        with self.assertRaises(PathNotFoundError,msg="video path defined is incorrect"):
            fl.extracting_frames(video_path,25)
        
        # to check the extracting_frames's methods behaviour if the video is in .mp4 format or not
        video_path = f'test_sample/incorrect_format/first/hello.wav'
        with self.assertRaises(FileNotFoundError, msg="Incorrect file format"):
            fl.extracting_frames(video_path, 25)

        # to check the behaviour of extracting_frames's if wrong number of the frames is provided 
        video_path= f'fit3162_fit3164/custom/data/video/M011/angry/001.mp4'
        with self.assertRaises(InvalidNumberofFrames,msg="video does not have given the right number of frames"):
            fl.extracting_frames(video_path,100)
        assert len(fl.extracting_frames(video_path,25))==25


    def test_generate_landmarks_frame(self):
        # to check if the landmark shape being generated from generate_landmarks_frame's method matches the requirement
        face_detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('fit3162_fit3164/shape_predictor_68_face_landmarks.dat')
        video_path= f'fit3162_fit3164/custom/data/video/M011/angry/001.mp4'
        assert fl.generate_landmarks_frame(fl.extracting_frames(video_path,1)[0]     , face_detector,predictor).shape==(136,)

    def test_facial_landmark(self):
      # to check if the shape of series of landmarks generated from facial_landmark's method amtches the requirement (25,136) where 25 represents to the number of frames and 136 refers to the num of landmark points for each of them
      video_path= f'fit3162_fit3164/custom/data/video/M011/angry/001.mp4'
      with patch("facial_landmark.generate_landmarks_frame") as generate:
            generate.return_value=np.array(136 *[0])
            landmarks=fl.facial_landmark(video_path,25)
            assert 25==  generate.call_count
      assert landmarks.shape==(25,136)


    def test_extracting_facial_landmarks(self):
        # to check the behaviour if the wrong directory is passed to extracting_facial_landmarks' method
        video_path=f'testing_sample/video/'
        with self.assertRaises(PathNotFoundError,msg="directory path is incorrect"):
            f2.extracting_facial_landmarks(video_path)

        # to check the behaviour if the wrong path name is passed to extracting_facial_landmarks' method
        video_path= f'fit3162_fit3164/custom/data/video/M011/furious'
        with self.assertRaises(PathNotFoundError, msg="Incorrect vieo path"):
            f2.extracting_facial_landmarks(video_path)

        #  to check the behaviour if the right path are passed and to ensure the files generated exist in their respective locations
        video_path=f'fit3162_fit3164/custom/data/video/M011'
        f2.extracting_facial_landmarks(video_path, True)
        val='001'
        assert os.path.exists(f'{LM_ENCODER_DATASET_LANDMARK_DIR}/{ACTOR}_angry_3_{val}/0.npy')==True
            

    def test_lm_pca(self):
        # different inputs cannot be checked as no argument is being passed to the function
        lm.lm_pca()        
        # to check if the necessary basic files for TESTACTOR are generated
        assert os.path.exists(f'{LANDMARK_BASICS}mean_68.npy')==True
        assert os.path.exists(f'{LANDMARK_BASICS}U_68.npy')==True
        assert os.path.exists(f'{LANDMARK_BASICS}val_{ACTOR}.pkl')==True
        assert os.path.exists(f'{LANDMARK_BASICS}train_{ACTOR}.pkl')==True
        
        # to check if the necessary mfcc for one of the file of M011 are generated
        val='001'
        assert os.path.exists(f'{LM_ENCODER_DATASET_MFCC_DIR}/{ACTOR}_angry_3_{val}/0.npy')==True    


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(FacialLandmarks)
    # This will run the test suite.
    unittest.TextTestRunner(verbosity=2).run(suite)