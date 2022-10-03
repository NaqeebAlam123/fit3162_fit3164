
import unittest
import audio2mfcc as a2m_convert
from unittest.mock import Mock, MagicMock, patch,call
import librosa
import io
import numpy as np
import os
from Exceptions_Classes import PathNotFoundError
import pickle
import re
class Audio2MfccTest(unittest.TestCase):

    # def setUp(self) -> None:
    #     self.mock_api = MagicMock()

    def test__audio2mfcc(self):
        path="D:\\Virtual box VMs\\fit3162_fit3164\custom\data\\audio\M030_wav\surprised\\030.wav"
        # The way most of the articles explained on how mfcc features is to have 20 ms window over the audio
        # for each mfcc feature and that window is displaced by 10 ms each time.So ,the number of mfcc features that will be generated will be
        # approximately equal to  (length of audio /10 ms)
        duration=librosa.get_duration(filename=path)
        expected_mfcc_features=(duration/0.01)
        save_path = "D:\\Virtual box VMs\\fit3162_fit3164\custom\data\\mfcc\M030\surprised_030"
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            a2m_convert._audio2mfcc(path, save_path,True)
            out=int(mock_stdout.getvalue().split("\n")[0])
            self.assertTrue(out<expected_mfcc_features+10 and out > expected_mfcc_features-10)
        # Since there are so many mfcc features, 28 sized window is rolled over all mfcc features to combine all
        # mfcc features inside the domain of the window into a single file.The window is pushed four positions ahead each time
        for i in range(int((out-28)/4)+1):
            assert os.path.exists(save_path+"\\"+str(i)+".npy")==True

        path = "D:\\Virtual box VMs\\fit3162_fit3164\custom\data\\audio\M030_wav\surprised\\?"
        with self.assertRaises(PathNotFoundError) as ex:
            a2m_convert._audio2mfcc(path, save_path,True)
        self.assertEqual(ex.exception.error_code,1)

        path = "D:\\Virtual box VMs\\fit3162_fit3164\custom\data\\audio\M030_wav\surprised\\030.wav"
        save_path=save_path = "D:\\Virtual box VMs\\fit3162_fit3164\custom\data\\mfcc\M030\?"
        with self.assertRaises(PathNotFoundError) as ex:
            a2m_convert._audio2mfcc(path, save_path,True)
        self.assertEqual(ex.exception.error_code, 2)

    def test_main(self):

        def find_num_files_dirs(list, path):
            total_files = 0
            for ele in list:
                files_in_dir = len(next(os.walk(os.path.join(path, ele)))[2])
                total_files = total_files + files_in_dir
            return total_files

        path="D:\\Virtual box VMs\\fit3162_fit3164\custom"
        with self.assertRaises(PathNotFoundError) as ex:
            a2m_convert.main(path,"")
        self.assertEqual(ex.exception.error_code, 3)

        path="D:\Virtual box VMs\\fit3162_fit3164\custom\data\\audio\M030"
        with self.assertRaises(PathNotFoundError) as ex:
            a2m_convert.main(path,"")
        self.assertEqual(ex.exception.error_code, 4)

        path="D:\Virtual box VMs\\fit3162_fit3164\custom\data\\audio\M030\\angry"
        with self.assertRaises(PathNotFoundError) as ex:
            a2m_convert.main(path,"")
        self.assertEqual(ex.exception.error_code, 5)

        path = "D:\Virtual box VMs\\fit3162_fit3164\custom\data\\audio\M030_wav"
        save_path="D:\Virtual box VMs\\fit3162_fit3164\custom\data\\mfcc\M030"
        sub_dirs=next(os.walk(path))[1]
        total_files=find_num_files_dirs(sub_dirs,path)

        sub_dirs = next(os.walk(save_path))[1]
        r=re.compile(".*00[1-9]$")
        val_dir_lst=list(filter(r.match,sub_dirs))
        train_dir_lst= list(set(sub_dirs).difference(val_dir_lst))

        number_of_val_files=find_num_files_dirs(val_dir_lst,save_path)
        number_of_train_files=find_num_files_dirs(train_dir_lst,save_path)

        with patch("audio2mfcc._audio2mfcc") as convert:
            a2m_convert.main(path,save_path,True)
            self.assertEqual(total_files,convert.call_count)

        with open('train_M030.pkl', 'rb') as f:
            self.assertEqual(len(pickle.load(f)),number_of_train_files)

        with open('val_M030.pkl', 'rb') as f:
            self.assertEqual(len(pickle.load(f)),number_of_val_files)











def main():


    suite = unittest.TestLoader().loadTestsFromTestCase(Audio2MfccTest)
    # This will run the test suite.
    unittest.TextTestRunner(verbosity=2).run(suite)