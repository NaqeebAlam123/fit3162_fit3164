import unittest
import facial_landmarks as fl
from Exceptions_Classes import *
import dlib
from unittest.mock import patch
import numpy as np
class FacialLandmarks(unittest.TestCase) :

    def setUp(self) -> None:
        pass

    def test_extracting_frames(self):
        video_path="D:\\readme.txt"
        with self.assertRaises(PathNotFoundError,msg="video path defined is incorrect"):
            fl.extracting_frames(video_path,25)
        video_path = "D:\\Virtual box VMs\\readme.txt"
        with self.assertRaises(FileNotFoundError, msg="Incorrect file or file not found"):
            fl.extracting_frames(video_path, 25)
        video_path= "../dataset_M030/video/angry/001.mp4"
        with self.assertRaises(InvalidNumberofFrames,msg="video does not have given number of frames"):
            fl.extracting_frames(video_path,100)

    def test_generate_landmarks_frame(self):
        face_detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(r"../../../shape_predictor_68_face_landmarks.dat")
        video_path = "../dataset_M030/video/angry/001.mp4"
        assert fl.generate_landmarks_frame(fl.extracting_frames(video_path,1)[0]     , face_detector,predictor).shape==(136,)

    def test_facial_landmark(self):
      video_path = "../dataset_M030/video/angry/001.mp4"
      with patch("facial_landmarks.generate_landmarks_frame") as generate:
            generate.return_value=np.array(136 *[0])
            landmarks=fl.facial_landmark(video_path,2)
            assert 2==  generate.call_count
      assert landmarks.shape==(2,136)


class ExtractingLandmarks(unittest.TestCase):

    def setUp(self) -> None:
        pass





if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(FacialLandmarks)
    # This will run the test suite.
    unittest.TextTestRunner(verbosity=2).run(suite)
