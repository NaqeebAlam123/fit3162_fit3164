import unittest
from dataload import SER_MFCC
from Exception_classes import *
import dlib
from unittest.mock import patch
import extracting_facial_landmarks as f2
import lm_pca as lm
import numpy as np
from constants import * 
import os
import pickle

class DataLoad(unittest.TestCase) :

    def setUp(self) -> None:
        pass



    def test_SER_MFCC(self):

        # ensure correct data being assigned to the self.train_data of the instance of SER_MFCC (whilst training and testing) 
        file = open(f'{EMOTION_NET_DATASET_DIR}basics/train_{ACTOR}.pkl', "rb")
        actual_train_data = pickle.load(file)
        file.close()
        
        result=SER_MFCC(EMOTION_NET_DATASET_DIR,'train')
        assert result.train_data==actual_train_data

        file = open(f'{EMOTION_NET_DATASET_DIR}basics/val_{ACTOR}.pkl', "rb")
        actual_val_data = pickle.load(file)
        file.close()
        
        result=SER_MFCC(EMOTION_NET_DATASET_DIR,'val')
        #print(result.train_data)
        assert result.train_data==actual_val_data  

        # ensure out of bound error is raised and handled properly when bigger index value is assigned to index while calling __getitem__ method of SER_MFCC object
        with self.assertRaises(IndexOutOfBoundError,msg="index is out of bound"):
           result.__getitem__(len(result.train_data))

        # ensure out of bound error is raised and handled properly when neg index value ssigned to index while calling __getitem__ method of SER_MFCC object
        with self.assertRaises(IndexOutOfBoundError,msg="index is out of bound"):
           result.__getitem__(-1)

        # check if the mfcc shape matches the requirement
        mfcc, label=result.__getitem__(len(result.train_data)-1)
        assert mfcc.shape==torch.Size([1,28,12])

        # check if the label matches the requirement (it should be zero as angry is denoted by '0' in MEAD)
        assert label==7

        # check if the __len__ method is returning the right size of train_data or not
        assert result.__len__()==len(result.train_data)

    def test_GET_MFCC(self):
        pass



if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(DataLoad)
    # This will run the test suite.
    unittest.TextTestRunner(verbosity=2).run(suite)