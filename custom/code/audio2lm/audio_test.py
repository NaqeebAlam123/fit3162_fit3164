import unittest
import audio2mfcc as a2m_convert
from unittest.mock import Mock, MagicMock, patch,call
import librosa
import io
import numpy as np
import os
from Exception_classes import PathNotFoundError
from mfcc_dtw import dtw_func 
import pickle
import re
from constants import *
from pathlib import Path

class AudioTest(unittest.TestCase):

    def setUp(self) -> None:
        pass


    def test__audio2mfcc(self):
        """
        purpose of testing:- 
        to generate mfcc features for particular audio samples

        inputs:- 
        1. audio_file which represents the path where .wav format file needs to be taken from
        2. save which represents the path where the generated mfcc need to be stored
        
        expected outputs:- 
        no returned value but the mfccs are generated from an audio and stored in the respective location. all of the exceptions are being handled properly.
        
        actual outputs observed:-
        all of the methods are doing their part according to the requirement as represented by the evidence provided in the docs
        """

        # # The way most of the articles explained on how mfcc features is to have 20 ms window over the audio
        # # for each mfcc feature and that window is displaced by 10 ms each time.So ,the number of mfcc features that will be generated will be
        # # approximately equal to  (length of audio /10 ms)
        path= f'{AUDIO_DATASET}/angry/001.wav'
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
        path= f'{AUDIO_DATASET}/angry/200.wav'
        with self.assertRaises(PathNotFoundError) as ex:
            a2m_convert.audio_to_mfcc_representation(path, save_path,True)
        self.assertEqual(ex.exception.error_code,1)

    def test_audio2mfcc_main(self):
        """
        purpose of testing:- 
        to generate mfcc features for the whole directory

        inputs:- 
        1. AUDIO_DATA refers to the directory where the audio file in .wav format are placed
        2. MFCC_OUTPUT refers to the directory where the generated mfccs need to be stored
        
        expected outputs:- 
        no returned value but all the generated mfcc for each of the audio sample is stored in the respective location

        actual outputs observed:-
        all of the methods are doing their part according to the requirement as represented by the evidence provided in the docs
        """
        
        # when the path being provided is not a directory but leads to a particular file
        path= f'{AUDIO_DATASET}/angry'
        save_path=f'{EMOTION_NET_DATASET_DIR}/generated_mfcc/'
        with self.assertRaises(PathNotFoundError) as ex:
            a2m_convert.main(path, save_path, True)
        self.assertEqual(ex.exception.error_code, 5)
        

        # when the audio in the path is of wrong format
        path='fit3162_fit3164/custom/data/audio/M030'
        save_path=f'{EMOTION_NET_DATASET_DIR}/generated_mfcc/'
        with self.assertRaises(PathNotFoundError) as ex:
            a2m_convert.main(path, save_path, True)
        self.assertEqual(ex.exception.error_code, 4)


        # when all the parameters are of the right format
        path= f'{AUDIO_DATASET}'
        save_path=f'{EMOTION_NET_DATASET_DIR}/generated_mfcc/'
        #a2m_convert.main(path, save_path, True)   -> generated mfccs will be provided for this case for faster running but can be commented out and ran on its own too


        # check if the basic files and generated mfccs exist
        assert os.path.exists(os.path.join(save_path, 'angry_001'))==True
        assert os.path.exists(f'{EMOTION_NET_DATASET_DIR}basics')==True
        mfcc_save_path=Path(os.path.join(save_path, 'angry_001'))
        basic_save_path=Path(f'{EMOTION_NET_DATASET_DIR}basics')
        assert len(list(mfcc_save_path.iterdir()))>0 # check if the subfiles are greater than zero for the particular audio file
        assert len(list(basic_save_path.iterdir()))==2 # check if the two required basic files are generated


    def test_mfcc_dtw(self):
        """
        purpose of testing:- 
        Dynamic Time Warping (DTW) algorithm is used to align MFCC 
        vectors of pairs of audios with the same content but different emotions. 
        These aligned audio samples can then be used as the inputs to the disentanglement network for cross-reconstruction.

        inputs:- 
        No inputs required

        expected outputs:- 
        aligned audio data and emotion length are generated and stored in the respective location

        actual outputs observed:-
        all of the methods are doing their part according to the requirement as represented by the evidence provided in the docs
        """
        
        
        # no input arguments are being provided so the existence of the new file and num of files being generated will be checked instead
        #dtw_func() -> generated aligned data and emotion length output will be provied for this case for faster running but can be commented out and ran on its own too
        assert os.path.exists(ALIGNED_AUDIO_DATA)==True
        assert os.path.exists(EMOTION_LENGTH_OUTPUT)==True
        aligned_audio_path=Path(ALIGNED_AUDIO_DATA)
        emotion_length_output=Path(EMOTION_LENGTH_OUTPUT)
        assert len(list(aligned_audio_path.iterdir()))==13
        assert len(list(emotion_length_output.iterdir()))==8
        
        # loading the pkl file data for one of the random samples from emotion length and aligned audio
        file = open(f'{EMOTION_LENGTH_OUTPUT}0/1.pkl', "rb")
        emotion_length_data = pickle.load(file)
        file.close()

        file = open(f'{ALIGNED_AUDIO_DATA}0/1.pkl', "rb")
        aligned_audio_data = pickle.load(file)
        file.close()

        # check if the pkl file data shape matches the requirement
        assert emotion_length_data.shape==(28,13)
        assert aligned_audio_data.shape==(324,13)
    

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(AudioTest)
    # This will run the test suite.
    unittest.TextTestRunner(verbosity=2).run(suite)