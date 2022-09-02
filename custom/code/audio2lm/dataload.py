import os
import random
import pickle
import numpy as np
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.functional import pad
import librosa
import time
import copy
from constants import LANDMARK_BASICS, LM_ENCODER_DATASET_MFCC_DIR, LM_ENCODER_DATASET_LANDMARK_DIR

DATAROOT = 'data/landmark/'


MEAD = {'angry':0, 'contempt':1, 'disgusted':2, 'fear':3, 'happy':4, 'neutral':5,
        'sad':6, 'surprised':7}
TRAIN_DIR = './train_M030.pkl'
VAL_DIR = './val_M030.pkl'

class SER_MFCC(data.Dataset):
    def __init__(self,
                 dataset_dir,train):

      #  self.data_path = dataset_dir
      #  file = open('/media/asus/840C73C4A631CC36/MEAD/SER_new/list.pkl', "rb") #'rb'-read binary file
      #  self.train_data = pickle.load(file)
      #  file.close()

        self.data_path = dataset_dir

        self.train = train
        if(self.train=='train'):
            file = open(TRAIN_DIR, "rb")
            self.train_data = pickle.load(file)
            file.close()
        if(self.train=='val'):
            file = open(VAL_DIR, "rb")
            self.train_data = pickle.load(file)
            file.close()

    def __getitem__(self, index):
        emotion = self.train_data[index].split('_')[0]
        label = torch.Tensor([MEAD[emotion]])

        mfcc_path = os.path.join(self.data_path ,  self.train_data[index])
        file = open(mfcc_path,'rb')
        mfcc = np.load(file)
        mfcc = mfcc[:,1:]
        mfcc = torch.FloatTensor(mfcc)
        mfcc=torch.unsqueeze(mfcc, 0)
        file.close()
        return mfcc, label

    def __len__(self):
        return len(self.train_data)


class GET_MFCC(data.Dataset):
    def __init__(self,
                 dataset_dir,phase):

        self.data_path = dataset_dir
        self.phase = phase

        self.emo_number = [0,1,2,3,4,5,6,7]
        # print(len(os.path.dirname(dataset_dir+'0/')), os.path.dirname(dataset_dir+'0/'))
        if phase == 'test':
            self.con_number = [i for i in range(len(os.listdir(dataset_dir+'0/')))]
        elif phase == 'train':
            self.con_number = [i for i in range(len(os.listdir(dataset_dir+'0/')))]



    def __getitem__(self, index): # build pseudo-training pairs
        #select name
        '''
        idx1, idx2,idx3, idx4 = np.random.choice(4,size=4)
        nidx1, nidx2,nidx3, nidx4 = self.name[idx1],self.name[idx2],self.name[idx3],self.name[idx4]

        idx1, idx2 = np.random.choice(4,size=2)
        oidx1, oidx2 = self.name[idx1],self.name[idx2]
        '''
        # select two emotions
        idx1, idx2 = np.random.choice(len(self.emo_number), size=2, replace=False)
        eidx1, eidx2 = self.emo_number[idx1], self.emo_number[idx2]
        # select three contents
        idx1, idx2, idx3 = np.random.choice(len(self.con_number), size=3, replace=True)

        cidx1, cidx2, cidx3= self.con_number[idx1],self.con_number[idx2], self.con_number[idx3]

        audio_path11 = os.path.join(self.data_path,str(eidx1)+'/'+str(cidx1)+'.pkl' )
  #      audio_path22 = os.path.join(self.data_path,str(eidx2)+'/'+str(cidx2)+'.pkl' )
        audio_path12 = os.path.join(self.data_path,str(eidx2)+'/'+str(cidx1)+'.pkl' )
  #      audio_path21 = os.path.join(self.data_path,str(eidx1)+'/'+str(cidx2)+'.pkl' )
        audio_path21 = os.path.join(self.data_path,str(eidx1)+'/'+str(cidx2)+'.pkl' )
        audio_path32 = os.path.join(self.data_path,str(eidx2)+'/'+str(cidx3)+'.pkl' )

        f=open(audio_path11,'rb')
        mfcc11=pickle.load(f)
        mfcc11 = torch.FloatTensor(mfcc11[:,1:])
        f.close()

#        f=open(audio_path22,'rb')
#        mfcc22=pickle.load(f)
#        mfcc22 = torch.FloatTensor(mfcc22[:,:12])
#        f.close()

        f=open(audio_path12,'rb')
        mfcc12=pickle.load(f)
        mfcc12 = torch.FloatTensor(mfcc12[:,1:])
        f.close()

        f=open(audio_path21,'rb')
        mfcc21=pickle.load(f)
        mfcc21 = torch.FloatTensor(mfcc21[:,1:])
        f.close()

        f=open(audio_path32,'rb')
        mfcc32=pickle.load(f)
        mfcc32 = torch.FloatTensor(mfcc32[:,1:])
        f.close()


        mfcc11=torch.unsqueeze(mfcc11, 0)
        mfcc21=torch.unsqueeze(mfcc21, 0)
        mfcc12=torch.unsqueeze(mfcc12, 0)
        mfcc32=torch.unsqueeze(mfcc32, 0)

        target11 = mfcc11.detach().clone()
#        target22 = mfcc22.detach().clone()

        target12 = mfcc12.detach().clone()
#        target21 = mfcc21.detach().clone()

        label1 = torch.tensor(eidx1).long()
        label2 = torch.tensor(eidx2).long()

        return {"input11": mfcc11, "target11": target11,
               "target21": target11, "target22": target12,
                "input12": mfcc12, "target12": target12,
                "label1": label1,  "label2": label2,
                "input21": mfcc21, "input32": mfcc32
              }


    def __len__(self):

       # return self.all_number * len(self.emo_number)
        return len(self.con_number) * len(self.emo_number)


class SMED_1D_lstm_landmark_pca(data.Dataset):
    def __init__(self,
                 dataset_dir,train = 'train'):

        self.num_frames = 16
        self.data_root = DATAROOT
      #  self.audio_root = '/media/asus/840C73C4A631CC36/MEAD/ATnet_emotion/dataset/MFCC'
        self.train = train

        file = open(f'{LANDMARK_BASICS}train_M030.pkl', "rb") #'rb'-read binary file
        self.train_data = pickle.load(file)
        print('train data', self.train_data)
        file.close()

        file = open(f'{LANDMARK_BASICS}val_M030.pkl', "rb") #'rb'-read binary file
        self.test_data = pickle.load(file)
        file.close()

        self.pca = torch.FloatTensor(np.load(f'{LANDMARK_BASICS}U_68.npy')[:, :16]).reshape(136,-1)
        self.mean = torch.FloatTensor(np.load(f'{LANDMARK_BASICS}mean_68.npy'))


    def __getitem__(self, index):
        if self.train == 'train':
            # ldmk loading
            data_folder = self.train_data[index]
            # print('data_folder', data_folder)
            lmark_path = os.path.join(LM_ENCODER_DATASET_LANDMARK_DIR, data_folder)
            audio_path = os.path.join(LM_ENCODER_DATASET_MFCC_DIR, data_folder )
            lmark = np.load(lmark_path)
            mfcc = np.load(audio_path)

            lmark = torch.FloatTensor(lmark)
            lmark = lmark - self.mean.expand_as(lmark)
            lmark = torch.mm(lmark, self.pca)

            # mfcc loading
            r = random.choice([x for x in range(3, 8)])
            # example_landmark = lmark[r, :]
            example_landmark = torch.FloatTensor(np.load(f'{LANDMARK_BASICS}mean_68.npy'))

            example_landmark = example_landmark - self.mean.expand_as(example_landmark)
            # example_landmark = example_landmark.reshape(-1,136)
            # example_landmark = torch.mm(example_landmark, self.pca)
            # import ipdb
            # ipdb.set_trace()
    
            # example_landmark = example_landmark.reshape(16)

            ## NOTE: 106-point
            # example_landmark = example_landmark.reshape(1,212)
            # example_landmark = torch.mm(example_landmark, self.pca)
    
            # example_landmark = example_landmark.reshape(16)

            example_mfcc = mfcc[r,:, 1:]
            mfccs = mfcc[r + 1: r + 17,:, 1:]

            mfccs = torch.FloatTensor(mfccs)
            landmark = lmark[r + 1: r + 17, :]

            landmark=landmark-example_landmark.expand_as(landmark)

            example_mfcc = torch.FloatTensor(example_mfcc)
            return example_landmark, example_mfcc, landmark, mfccs

        if self.train == 'test':
            # ldmk loading
            data_folder = self.test_data[index]
            lmark_path = os.path.join(LM_ENCODER_DATASET_LANDMARK_DIR, data_folder)
            audio_path = os.path.join(LM_ENCODER_DATASET_MFCC_DIR, data_folder )
            lmark = np.load(lmark_path)
            mfcc = np.load(audio_path)

            lmark = torch.FloatTensor(lmark)
            lmark = lmark - self.mean.expand_as(lmark)
            lmark = torch.mm(lmark, self.pca)

            # mfcc loading


            r = random.choice([x for x in range(3, 8)])
          #  example_landmark = lmark[r, :]
            example_landmark = torch.FloatTensor(np.load(f'{LANDMARK_BASICS}mean_68.npy'))
            example_landmark = example_landmark - self.mean.expand_as(example_landmark)
            example_landmark = example_landmark.reshape(1,136)
            example_landmark = torch.mm(example_landmark, self.pca)
            example_landmark = example_landmark.reshape(16)

            example_mfcc = mfcc[r,:, 1:]
            mfccs = mfcc[r + 1: r + 17,:, 1:]

            mfccs = torch.FloatTensor(mfccs)
            landmark = lmark[r + 1: r + 17, :]

            landmark=landmark-example_landmark.expand_as(landmark)

            example_mfcc = torch.FloatTensor(example_mfcc)
            return example_landmark, example_mfcc, landmark, mfccs

    def __len__(self):
        if self.train == 'train':
            return len(self.train_data)
        if self.train == 'test':
            return len(self.test_data)
