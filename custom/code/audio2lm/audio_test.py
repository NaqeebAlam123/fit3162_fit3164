import unittest
import audio2mfcc as a2m_convert
from unittest.mock import Mock, MagicMock, patch,call
import librosa
import io
import numpy as np
import os
from Exception_classes import PathNotFoundError
import pickle
import re
from constants import *
from pathlib import Path

class AudioTest(unittest.TestCase):

    def setUp(self) -> None:
        pass


    def test__audio2mfcc(self):
        
        # # The way most of the articles explained on how mfcc features is to have 20 ms window over the audio
        # # for each mfcc feature and that window is displaced by 10 ms each time.So ,the number of mfcc features that will be generated will be
        # # approximately equal to  (length of audio /10 ms)
        path= f'test_sample/correct_audio/TEST/angry/001.wav'
        duration=librosa.get_duration(filename=path)
        expected_mfcc_features=(duration/0.01)
        save_path=f'{EMOTION_NET_DATASET_DIR}/generated_mfcc/angry_001'
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            a2m_convert.audio_to_mfcc_representation(path, save_path,True)
            out=int(mock_stdout.getvalue().split("\n")[0])
            self.assertTrue(out<expected_mfcc_features+10 and out > expected_mfcc_features-10)

        # Since there are so many mfcc features, 28 sized window is rolled over all mfcc features to combine all
        # mfcc features inside the domain of the window into a single file.The window is pushed four positions ahead each time
        for i in range(int((out-28)/4)+1):
            assert os.path.exists(save_path+"\\"+str(i)+".npy")==True

        # checking with wrong audio file being supplied
        path = f'test_sample/correct_audio/TEST/angry/010.wav'
        with self.assertRaises(PathNotFoundError) as ex:
            a2m_convert.audio_to_mfcc_representation(path, save_path,True)
        self.assertEqual(ex.exception.error_code,1)

    def test_audio2mfcc_main(self):
        # when the path being provided is not a directory but leads to a particular file
        path='test_sample/correct_audio/TEST/angry'
        save_path=f'{EMOTION_NET_DATASET_DIR}/generated_mfcc/'
        with self.assertRaises(PathNotFoundError) as ex:
            a2m_convert.main(path, save_path, True)
        self.assertEqual(ex.exception.error_code, 5)
        

        # when the audio in the path is of wrong format
        path='test_sample/incorrect_audio/TEST'
        save_path=f'{EMOTION_NET_DATASET_DIR}/generated_mfcc/'
        with self.assertRaises(PathNotFoundError) as ex:
            a2m_convert.main(path, save_path, True)
        self.assertEqual(ex.exception.error_code, 4)


        # when all the parameters are of the right format
        path='test_sample/correct_audio/TEST'
        save_path=f'{EMOTION_NET_DATASET_DIR}/generated_mfcc/'
        a2m_convert.main(path, save_path, True)


         # check if the basic files and generated mfccs exist
        assert os.path.exists(os.path.join(save_path, 'angry_001'))==True
        assert os.path.exists(f'{EMOTION_NET_DATASET_DIR}basics')==True
        mfcc_save_path=Path(os.path.join(save_path, 'angry_001'))
        basic_save_path=Path(f'{EMOTION_NET_DATASET_DIR}basics')
        assert len(list(mfcc_save_path.iterdir()))>0 # check if the subfiles are greater than zero for the particular audio file
        assert len(list(basic_save_path.iterdir()))==2 # check if the two required basic files are generated



if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(AudioTest)
    # This will run the test suite.
    unittest.TextTestRunner(verbosity=2).run(suite)
