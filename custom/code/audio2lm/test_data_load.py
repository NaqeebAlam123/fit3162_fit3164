import unittest
from dataload import GET_MFCC, SER_MFCC, SMED_1D_lstm_landmark_pca
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
        """
        purpose of testing:- 
        all the methods of SER_MFCC (which caters with returning testing and training dataset to the EmotioNet)'s all instance variables 
        and return values have the correct data and is in the required shape (if applicable)
        method of testing:- make use of python unittest framework and assert statements
        
        inputs:- 
        1. path of mfcc features of the particular "ACTOR"  
        2. train is either equivalent to "train" or "test" dependending if the EmotionNet is requiring dataset for training or validation
        
        expected outputs:- 
        _init_ method should assign path of particular mfcc features to train_data
        __getitem__ method should return the mfcc features along with the label (depicting the emotion) and if supplied 
        with wrong index then raise exceptions
        __len__ method should return the size of train_data
        
        actual outputs observed:-
        all of the methods are doing their part according to the requirement as represented by the evidence provided in the docs
        """

        # ensure PathNotFound error is raised and handled properly when invalid path is assigned to dataset_dir
        with self.assertRaises(PathNotFoundError,msg="path doesn't exist"):
            SER_MFCC('/result', 'train')

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
        """
        purpose of testing:- 
        all the methods of GET_MFCC (which caters with returning testing and training dataset to the AutoEncoder2x)'s 
        all instance variables and return values have the correct data and is in the required shape (if applicable)

        inputs:- 
        1. path of emotion length of the particular "ACTOR"  
        
        expected outputs:- 
        _init__ method should assign path of particular emotion length to data_path, MEAD values representing different emotion to emo_number, subfolder names to the con_number
        __getitem__ is used for building the pseudo-training pairs 
        __len__ method should return the total number of emotion length's subfolders
        
        actual outputs observed:-
        all of the methods are doing their part according to the requirement as represented by the evidence provided in the docs
        """

        #GET_MFCC('/result', 'train')
        # ensure PathNotFound error is raised and handled properly when invalid path is assigned to dataset_dir
        with self.assertRaises(PathNotFoundError,msg="path doesn't exist"):
            GET_MFCC('/result', 'train')


        result=GET_MFCC(AUTOENCODER_2X_DATASET_DIR,'train')
        # ensuring correct values for __int__ method of GET_MFCC are set 
        assert result.data_path==AUTOENCODER_2X_DATASET_DIR
        assert result.emo_number==[0,1,2,3,4,5,6,7]
        assert result.con_number==[i for i in range(len(os.listdir(AUTOENCODER_2X_DATASET_DIR+'0/')))]
        
        dict_item=result.__getitem__(1)
       
        # ensure the inputs along with target values are returned
        assert len(dict_item)==10
        assert len(dict_item.get('input11'))==1
        assert len(dict_item.get('target11'))==1
        assert len(dict_item.get('target21'))==1
        assert len(dict_item.get('target22'))==1
        assert len(dict_item.get('input12'))==1
        assert len(dict_item.get('target12'))==1
        assert len(dict_item.get('input21'))==1
        assert len(dict_item.get('input32'))==1

        # ensure the len function return thr right result 
        assert len(result.emo_number)*len(result.con_number)==result.__len__()

    def test_SMED_1D_lstm_landmark_pca(self):
        """
        purpose of testing:- 
        all the methods of SMED_1D_lstm_landmark_pca (which caters with returning testing and training dataset to the AT_Emotion)'s 
        all instance variables and return values have the correct data and is in the required shape (if applicable)

        inputs:- 
        1. path of emotion length of the particular "ACTOR"  
        
        expected outputs:- 
        _init_ method should assign content of train pkl basic folder of ACTOR to train_data, 
        assign content of val pkl basic folder of ACTOR to test_data, 
        assign content of U_68.npy of ACTOR to pca, assign content of mean_68.npy to mean
        __getitem__ is used for returning the example landmarks and mfccs along with the landmarks and mfccs
        __len__ method should return the length of required data (train or test)
        
        actual outputs observed:-
        all of the methods are doing their part according to the requirement as represented by the evidence provided in the docs
        """
        # creating an instance of SMED_1D_lstm_landmark_pca for training set
        result=SMED_1D_lstm_landmark_pca(LM_ENCODER_DATASET_LANDMARK_DIR, "train")

        # ensure out of bound error is raised and handled properly when bigger index value is assigned to index while calling __getitem__ method of SMED_1D_lstm_landmark_pca object
        with self.assertRaises(IndexOutOfBoundError,msg="index is out of bound"):
           result.__getitem__(len(result.train_data))

        # ensure out of bound error is raised and handled properly when neg index value ssigned to index while calling __getitem__ method of SMED_1D_lstm_landmark_pca object
        with self.assertRaises(IndexOutOfBoundError,msg="index is out of bound"):
           result.__getitem__(-1)

        
        # loading the actual train data
        file = open(f'{LANDMARK_BASICS}train_{ACTOR}.pkl', "rb")
        train_data = pickle.load(file)
        file.close()

        # check if the training set matches the actual set 
        self.assertEqual(train_data, result.train_data)

        # check if the example mfcc and landmark shape matches the requirement
        self.assertEqual(result.__getitem__(len(result)-1)[0].shape, torch.Size([16]))
        self.assertEqual(result.__getitem__(len(result)-1)[1].shape, torch.Size([28, 12]) )
        
        
        # check if the mfcc and landmark shape matches the requirement
        self.assertEqual(result.__getitem__(len(result)-1)[2].shape, torch.Size([16, 16]))
        self.assertEqual(result.__getitem__(len(result)-1)[3].shape, torch.Size([16, 28, 12]))
        
        # ensure the len function return the right size for train data
        self.assertEqual(result.__len__(),len(result.train_data))
        
        # creating an instance of SMED_1D_lstm_landmark_pca for testing set
        result=SMED_1D_lstm_landmark_pca(LM_ENCODER_DATASET_LANDMARK_DIR, 'test')

        # loading the test data
        file = open(f'{LANDMARK_BASICS}val_{ACTOR}.pkl', "rb")
        test_data = pickle.load(file)
        file.close()

        # check if the testing set matches the actual set 
        self.assertEqual(test_data, result.test_data)

        # ensure the len function return the right size for train data
        self.assertEqual(result.__len__(),len(result.test_data))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(DataLoad)
    # This will run the test suite.
    unittest.TextTestRunner(verbosity=2).run(suite)