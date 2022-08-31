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

        print(os.path.dirname(self.data_path+'0/'))
        print(len(self.con_number))
        print(audio_path11)
        print(audio_path12)
        print(audio_path21)
        print(audio_path32)

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
        self.data_root = '../dataset_M030/'
      #  self.audio_root = '/media/asus/840C73C4A631CC36/MEAD/ATnet_emotion/dataset/MFCC'
        self.train = train

        file = open('landmark/basics/train_M030.pkl', "rb") #'rb'-read binary file
        self.train_data = pickle.load(file)
        print('train data', self.train_data)
        file.close()

        file = open('landmark/basics/val_M030.pkl', "rb") #'rb'-read binary file
        self.test_data = pickle.load(file)
        file.close()

        self.pca = torch.FloatTensor(np.load('landmark/basics/U_106.npy')[:, :16])
        self.mean = torch.FloatTensor(np.load('landmark/basics/mean_106.npy'))


    def __getitem__(self, index):
        if self.train == 'train':
            # ldmk loading
            data_folder = self.train_data[index]
            print('data_folder', data_folder)
            lmark_path = os.path.join(self.data_root, 'landmark', data_folder)
            audio_path = os.path.join(self.data_root, 'mfcc', data_folder )
            lmark = np.load(lmark_path)
            mfcc = np.load(audio_path)

            lmark = torch.FloatTensor(lmark)
            lmark = lmark - self.mean.expand_as(lmark)
            lmark = torch.mm(lmark, self.pca)

            # mfcc loading


            r = random.choice([x for x in range(3, 8)])
          #  example_landmark = lmark[r, :]
            example_landmark = torch.FloatTensor(np.load('landmark/basics/mean_106.npy'))

            example_landmark = example_landmark - self.mean.expand_as(example_landmark)
            example_landmark = example_landmark.reshape(1,212)
            example_landmark = torch.mm(example_landmark, self.pca)
            example_landmark = example_landmark.reshape(16)


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
            lmark_path = os.path.join(self.data_root, 'landmark', data_folder)
            audio_path = os.path.join(self.data_root, 'mfcc', data_folder )
            lmark = np.load(lmark_path)
            mfcc = np.load(audio_path)

            lmark = torch.FloatTensor(lmark)
            lmark = lmark - self.mean.expand_as(lmark)
            lmark = torch.mm(lmark, self.pca)

            # mfcc loading


            r = random.choice([x for x in range(3, 8)])
          #  example_landmark = lmark[r, :]
            example_landmark = torch.FloatTensor(np.load('landmark/basics/mean_106.npy'))
            example_landmark = example_landmark - self.mean.expand_as(example_landmark)
            example_landmark = example_landmark.reshape(1,212)
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


class SMEDataset1D_lstm_gt(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 output_shape=[128, 128],
                 train='train'):
        self.train = train
        self.dataset_dir = dataset_dir
        self.output_shape = tuple(output_shape)

        if not len(output_shape) in [2, 3]:
            raise ValueError("[*] output_shape must be [H,W] or [C,H,W]")

        if self.train == 'train':
            self.video_root = '/mnt/lustrenew/share/zhuhao_777/Video25'
            self.videos = os.listdir(self.video_root)
            self.videos.sort()
            self.data_root = '../dataset/Video'
            self.folders = os.listdir(self.data_root)
            self.folders.sort()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
        if self.train == 'train':

            # load right img
            image_folder = self.videos[index] + '/crop_256/frames'
            image_folder_path = os.path.join(self.video_root, image_folder)
            image_data = os.listdir(image_folder_path)
            image_data.sort()

            current_frame_id = np.random.randint(0, len(image_data)-16)
            right_img = torch.FloatTensor(16, 3, self.output_shape[0], self.output_shape[1])
            for jj in range(16):
                this_frame = current_frame_id + jj
                image_path = image_folder_path + '/' + image_data[0][:-10] + '%06d.jpg' % this_frame
                im = cv2.imread(image_path)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = cv2.resize(im, self.output_shape)
                im = self.transform(im)
                right_img[jj, :, :, :] = torch.FloatTensor(im)


            data_folder = self.folders[index]
            data_path = os.path.join(self.data_root, data_folder)
            ldmk_path = os.path.join(data_path, 'landmark.txt')
            ldmk_data = self.parameter_reader(ldmk_path)


            right_landmark = torch.FloatTensor(ldmk_data[current_frame_id : current_frame_id + 16])

            right_landmark = right_landmark.reshape(16, 136)

            r = random.choice(
                [x for x in range(1, 30)])
            example_path = image_folder_path + '/' + image_data[0][:-10] + '%06d.jpg' % r
            example_landmark = ldmk_data[r]

            example_landmark = torch.FloatTensor(example_landmark).reshape(-1)

            example_img = cv2.imread(example_path)
            example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
            example_img = cv2.resize(example_img, self.output_shape)
            example_img = self.transform(example_img)
            # print (right_landmark.size())

            return example_img, example_landmark, right_img, right_landmark


        elif self.train == 'test':
            # load righ img
            image_path = '../dataset/regions/' + self.test_data[index][0]
            landmark_path = '../dataset/landmark1d/' + self.test_data[index][0][:-8] + '.npy'
            current_frame_id = self.test_data[index][1]
            right_img = torch.FloatTensor(16, 3, self.output_shape[0], self.output_shape[1])
            for jj in range(16):
                this_frame = current_frame_id + jj
                image_path = '../dataset/regions/' + self.test_data[index][0][:-7] + '%03d.jpg' % this_frame
                im = cv2.imread(image_path)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = cv2.resize(im, self.output_shape)
                im = self.transform(im)
                right_img[jj, :, :, :] = torch.FloatTensor(im)

            landmark = np.load(landmark_path) * 5.0

            right_landmark = landmark[self.test_data[index][1] - 1: self.test_data[index][1] + 15]

            right_landmark = torch.FloatTensor(right_landmark.reshape(16, 136))
            r = random.choice(
                [x for x in range(1, 30)])
            r = current_frame_id
            example_path = image_path[:-8] + '_%03d.jpg' % r
            example_landmark = landmark[r - 1]

            example_landmark = torch.FloatTensor(example_landmark.reshape(-1))

            example_img = cv2.imread(example_path)
            example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
            example_img = cv2.resize(example_img, self.output_shape)
            example_img = self.transform(example_img)
            # print (right_landmark.size())

            return example_img, example_landmark, right_img, right_landmark

    def __len__(self):
        if self.train == 'train':
            return len(self.videos)
        elif self.train == 'test':
            return len(self.test_data)
        else:
            return len(self.demo_data)

    def parameter_reader(self, flist):
        parameter_list = []
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                parameters = line.strip().split()
                # for i in range(len(parameters)):
                #     parameters[i] = float(parameters[i])
                params = [float(i) for i in parameters]
                parameter_list.append(params)
        return parameter_list


class LRW_1D_lstm_landmark_pca(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 train='train'):
        self.train = train
        self.num_frames = 16
        self.lmark_root_path = '../dataset/landmark1d'
        self.pca = torch.FloatTensor(np.load('landmark/basics/U_lrw1.npy')[:,:6] )
        self.mean = torch.FloatTensor(np.load('landmark/basics/mean_lrw1.npy'))
        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "lmark_train.pkl"), "rb")
            self.train_data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "lmark_test.pkl"), "rb")
            self.test_data = pickle.load(_file)
            _file.close()
        elif self.train =='demo' :
            _file = open(os.path.join(dataset_dir, "img_demo.pkl"), "rb")
            self.demo_data = pickle.load(_file)
            _file.close()


    def __getitem__(self, index):
        if self.train=='train':
            lmark_path = os.path.join(self.lmark_root_path , self.train_data[index][0] , self.train_data[index][1],self.train_data[index][2], self.train_data[index][2] + '.npy')
            mfcc_path = os.path.join('../dataset/mfcc/',  self.train_data[index][0],  self.train_data[index][1],  self.train_data[index][2] + '.npy')

            lmark = np.load(lmark_path) * 5.0
            lmark = torch.FloatTensor(lmark)
            lmark = lmark - self.mean.expand_as(lmark)
            lmark = torch.mm(lmark,self.pca)

            mfcc = np.load(mfcc_path)

            r = random.choice(
                [x for x in range(3,8)])
            example_landmark =lmark[r,:]
            example_mfcc = mfcc[(r -3) * 4 : (r + 4) * 4, 1 :]
            mfccs = []
            for ind in range(1,17):
                t_mfcc =mfcc[(r + ind - 3)*4: (r + ind + 4)*4, 1:]
                t_mfcc = torch.FloatTensor(t_mfcc)
                mfccs.append(t_mfcc)
            mfccs = torch.stack(mfccs, dim = 0)
            landmark  =lmark[r+1 : r + 17,:]
            example_mfcc = torch.FloatTensor(example_mfcc)
            return example_landmark, example_mfcc, landmark, mfccs
        if self.train=='test':
            lmark_path = os.path.join(self.lmark_root_path , self.test_data[index][0] , self.test_data[index][1],self.test_data[index][2], self.test_data[index][2] + '.npy')
            mfcc_path = os.path.join('../dataset/mfcc/',  self.test_data[index][0],  self.test_data[index][1],  self.test_data[index][2] + '.npy')

            lmark = np.load(lmark_path) * 5.0
            lmark = torch.FloatTensor(lmark)
            lmark = lmark - self.mean.expand_as(lmark)
            lmark = torch.mm(lmark,self.pca)

            mfcc = np.load(mfcc_path)

            r = random.choice(
                [x for x in range(3,8)])
            example_landmark =lmark[r,:]
            example_mfcc = mfcc[(r -3) * 4 : (r + 4) * 4, 1 :]
            mfccs = []
            for ind in range(1,17):
                t_mfcc =mfcc[(r + ind - 3)*4: (r + ind + 4)*4, 1:]
                t_mfcc = torch.FloatTensor(t_mfcc)
                mfccs.append(t_mfcc)
            mfccs = torch.stack(mfccs, dim = 0)
            landmark  =lmark[r+1 : r + 17,:]
            example_mfcc = torch.FloatTensor(example_mfcc)
            return example_landmark, example_mfcc, landmark, mfccs

    def __len__(self):
        if self.train=='train':
            return len(self.train_data)
        elif self.train=='test':
            return len(self.test_data)
        # else:
        #     pas
class LRW_1D_single_landmark_pca(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 train='train'):
        self.train = train
        self.num_frames = 16
        self.lmark_root_path = '../dataset/landmark1d'
        self.audio_root_path = '../dataset/audio'
        self.pca = torch.FloatTensor(np.load('landmark/basics/U_lrw1.npy')[:,:6] )
        self.mean = torch.FloatTensor(np.load('landmark/basics/mean_lrw1.npy'))

        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "lmark_train.pkl"), "rb")
            self.train_data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "lmark_test.pkl"), "rb")
            self.test_data = pickle.load(_file)
            _file.close()
        elif self.train =='demo' :
            _file = open(os.path.join(dataset_dir, "img_demo.pkl"), "rb")
            self.demo_data = pickle.load(_file)
            _file.close()




    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
        if self.train=='train':
            lmark_path = os.path.join(self.lmark_root_path , self.train_data[index][0] , self.train_data[index][1],self.train_data[index][2], self.train_data[index][2] + '.npy')
            mfcc_path = os.path.join('../dataset/mfcc/',  self.train_data[index][0],  self.train_data[index][1],  self.train_data[index][2] + '.npy')

            lmark = np.load(lmark_path)

            lmark = torch.FloatTensor(lmark) * 5.0
            lmark = lmark - self.mean.expand_as(lmark)
            lmark = torch.mm(lmark,self.pca)

            mfcc = np.load(mfcc_path)

            r = random.choice(
                [x for x in range(3,25)])
            example_landmark =lmark[r,:]
            example_mfcc = mfcc[(r -3) * 4 : (r + 4) * 4, 1 :]

            while True:
                current_frame_id = random.choice(
                    [x for x in range(3,25)])
                if current_frame_id != r:
                    break
            t_mfcc =mfcc[( current_frame_id - 3)*4: (current_frame_id + 4)*4, 1:]
            t_mfcc = torch.FloatTensor(t_mfcc)
            landmark  =lmark[current_frame_id , :]
            example_landmark = torch.FloatTensor(example_landmark)
            example_mfcc = torch.FloatTensor(example_mfcc)
            landmark = torch.FloatTensor(landmark)
            landmark = landmark
            return example_landmark, example_mfcc, landmark, t_mfcc
        if self.train=='test':
            mfcc_path = os.path.join('../dataset/mfcc/',  self.test_data[index][0],  self.test_data[index][1],  self.test_data[index][2] + '.npy')
            lmark_path = os.path.join(self.lmark_root_path , self.test_data[index][0] , self.test_data[index][1],self.test_data[index][2], self.test_data[index][2] + '.npy')
            lmark = np.load(lmark_path)
            mfcc = np.load(mfcc_path)
            example_landmark =lmark[3,:]
            example_mfcc = mfcc[0 : 7 * 4, 1 :]
            r =3
            ind = self.test_data[index][3]

            t_mfcc =mfcc[(r + ind - 3)*4: (r + ind + 4)*4, 1:]
            t_mfcc = torch.FloatTensor(t_mfcc)
            landmark  =lmark[r+ ind,:]
            # example_audio = torch.FloatTensor(example_audio)
            example_mfcc = torch.FloatTensor(example_mfcc)
            # audio = torch.FloatTensor(audio)
            # mfccs = torch.FloatTensor(mfccs)
            landmark = torch.FloatTensor(landmark)
            # landmark = self.transform(landmark)
            landmark = landmark * 5.0
            example_landmark = torch.FloatTensor(example_landmark).view(1,-1)
            example_landmark = example_landmark - self.mean.expand_as(example_landmark)
            example_landmark = torch.mm(example_landmark,self.pca)

            return example_landmark[0], example_mfcc, landmark, t_mfcc

    def __len__(self):
        if self.train=='train':
            return len(self.train_data)
        elif self.train=='test':
            return len(self.test_data)
        else:
            pass

class LRWdataset1D_single_gt(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 output_shape=[128, 128],
                 train='train'):
        self.train = train
        self.dataset_dir = dataset_dir
        self.output_shape = tuple(output_shape)

        if not len(output_shape) in [2, 3]:
            raise ValueError("[*] output_shape must be [H,W] or [C,H,W]")

        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "new_img_full_gt_train.pkl"), "rb")
            self.train_data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "new_img_full_gt_test.pkl"), "rb")
            self.test_data = pickle.load(_file)
            _file.close()
        elif self.train =='demo' :
            _file = open(os.path.join(dataset_dir, "new_img_full_gt_demo.pkl"), "rb")
            self.demo_data = pickle.load(_file)
            _file.close()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
        if self.train=='train':

                #load righ img
                image_path = '../dataset/regions/' +  self.train_data[index][0]
                landmark_path = '../dataset/landmark1d/' + self.train_data[index][0][:-8] + '.npy'

                landmark = np.load(landmark_path) * 5.0

                right_landmark = landmark[self.train_data[index][1] - 1]
                right_landmark = torch.FloatTensor(right_landmark.reshape(-1))

                im = cv2.imread(image_path)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = cv2.resize(im, self.output_shape)
                im = self.transform(im)
                right_img = torch.FloatTensor(im)

                r = random.choice(
                    [x for x in range(1,30)])
                example_path =   image_path[:-8] + '_%03d.jpg'%r
                example_landmark = landmark[r - 1]
                example_landmark = torch.FloatTensor(example_landmark.reshape(-1))

                example_img = cv2.imread(example_path)
                example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
                example_img = cv2.resize(example_img, self.output_shape)
                example_img = self.transform(example_img)

                return example_img, example_landmark, right_img,right_landmark

        elif self.train =='test':
            # try:
                #load righ img
            image_path = '../dataset/regions/' +  self.test_data[index][0]
            landmark_path = '../dataset/landmark1d/' + self.test_data[index][0][:-8] + '.npy'
            landmark = np.load(landmark_path) * 5.0
            right_landmark = landmark[self.test_data[index][1] - 1]

            right_landmark = torch.FloatTensor(right_landmark.reshape(-1))

            im = cv2.imread(image_path)

            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, self.output_shape)
            im = self.transform(im)
            right_img = torch.FloatTensor(im)

            example_path =   '../image/150_region.jpg'
            example_landmark = np.load('../image/musk1.npy')

            example_landmark = torch.FloatTensor(example_landmark.reshape(-1)) * 5.0

            example_img = cv2.imread(example_path)
            example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
            example_img = cv2.resize(example_img, self.output_shape)
            example_img = self.transform(example_img)

            return example_img, example_landmark, right_img,right_landmark


class LRWdataset1D_lstm_gt(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 output_shape=[128, 128],
                 train='train'):
        self.train = train
        self.dataset_dir = dataset_dir
        self.output_shape = tuple(output_shape)

        if not len(output_shape) in [2, 3]:
            raise ValueError("[*] output_shape must be [H,W] or [C,H,W]")

        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "new_16_full_gt_train.pkl"), "rb")
            self.train_data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "new_16_full_gt_test.pkl"), "rb")
            self.test_data = pickle.load(_file)
            _file.close()
        elif self.train =='demo' :
            _file = open(os.path.join(dataset_dir, "new_16_full_gt_demo.pkl"), "rb")
            self.demo_data = pickle.load(_file)
            _file.close()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
        if self.train=='train':

                #load righ img
                image_path = '../dataset/regions/' +  self.train_data[index][0]
                landmark_path = '../dataset/landmark1d/' + self.train_data[index][0][:-8] + '.npy'
                current_frame_id =self.train_data[index][1]
                right_img = torch.FloatTensor(16,3,self.output_shape[0],self.output_shape[1])
                for jj in range(16):
                    this_frame = current_frame_id + jj
                    image_path =  '../dataset/regions/' +  self.train_data[index][0][:-7] + '%03d.jpg'%this_frame
                    im = cv2.imread(image_path)
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    im = cv2.resize(im, self.output_shape)
                    im = self.transform(im)
                    right_img[jj,:,:,:] = torch.FloatTensor(im)



                landmark = np.load(landmark_path) * 5.0

                right_landmark = landmark[self.train_data[index][1] - 1 : self.train_data[index][1] + 15  ]

                right_landmark = torch.FloatTensor(right_landmark.reshape(16,136))


                r = random.choice(
                    [x for x in range(1,30)])
                example_path =   image_path[:-8] + '_%03d.jpg'%r
                example_landmark = landmark[r - 1]

                example_landmark = torch.FloatTensor(example_landmark.reshape(-1))

                example_img = cv2.imread(example_path)
                example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
                example_img = cv2.resize(example_img, self.output_shape)
                example_img = self.transform(example_img)
                # print (right_landmark.size())

                return example_img, example_landmark, right_img,right_landmark


        elif self.train =='test':
            #load righ img
                image_path = '../dataset/regions/' +  self.test_data[index][0]
                landmark_path = '../dataset/landmark1d/' + self.test_data[index][0][:-8] + '.npy'
                current_frame_id =self.test_data[index][1]
                right_img = torch.FloatTensor(16,3,self.output_shape[0],self.output_shape[1])
                for jj in range(16):
                    this_frame = current_frame_id + jj
                    image_path =  '../dataset/regions/' +  self.test_data[index][0][:-7] + '%03d.jpg'%this_frame
                    im = cv2.imread(image_path)
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    im = cv2.resize(im, self.output_shape)
                    im = self.transform(im)
                    right_img[jj,:,:,:] = torch.FloatTensor(im)

                landmark = np.load(landmark_path) * 5.0

                right_landmark = landmark[self.test_data[index][1] - 1 : self.test_data[index][1] + 15  ]

                right_landmark = torch.FloatTensor(right_landmark.reshape(16,136))
                r = random.choice(
                    [x for x in range(1,30)])
                r = current_frame_id
                example_path =   image_path[:-8] + '_%03d.jpg'%r
                example_landmark = landmark[r - 1]

                example_landmark = torch.FloatTensor(example_landmark.reshape(-1))

                example_img = cv2.imread(example_path)
                example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
                example_img = cv2.resize(example_img, self.output_shape)
                example_img = self.transform(example_img)
                # print (right_landmark.size())

                return example_img, example_landmark, right_img,right_landmark


    def __len__(self):
        if self.train=='train':
            return len(self.train_data)
        elif self.train=='test':
            return len(self.test_data)
        else:
            return len(self.demo_data)


class LRWdataset1D_single(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 output_shape=[128, 128],
                 train='train'):
        self.train = train
        self.dataset_dir = dataset_dir
        self.output_shape = tuple(output_shape)

        if not len(output_shape) in [2, 3]:
            raise ValueError("[*] output_shape must be [H,W] or [C,H,W]")

        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "new_img_small_train.pkl"), "rb")
            self.train_data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "new_img_small_test.pkl"), "rb")
            self.test_data = pickle.load(_file)
            _file.close()
        elif self.train =='demo' :
            _file = open(os.path.join(dataset_dir, "img_demo.pkl"), "rb")
            self.demo_data = pickle.load(_file)
            _file.close()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
        if self.train=='train':
            while True:
                # try:
                    #load righ img
                image_path = self.train_data[index][0]
                landmark_path = self.train_data[index][1]
                landmark = np.load(landmark_path)

                right_landmark = landmark[self.train_data[index][2]]
                tp = ( np.dot(right_landmark.reshape(1,6), EIGVECS))[0,:].reshape(68,3)
                tp = tp[:,:-1].reshape(-1)
                right_landmark = torch.FloatTensor(tp)
                im = cv2.imread(image_path)
                if im is None:
                    raise IOError
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = cv2.resize(im, self.output_shape)
                im = self.transform(im)
                right_img = torch.FloatTensor(im)

                r = random.choice(
                        [x for x in range(1,30)])
                example_path = image_path[:-8] + '_%03d.jpg'%r
                example_landmark = landmark[r]


                tp = ( np.dot(example_landmark.reshape(1,6), EIGVECS))[0,:].reshape(68,3)
                tp = tp[:,:-1].reshape(-1)


                example_landmark = torch.FloatTensor(tp)

                example_img = cv2.imread(example_path)
                if example_img is None:
                    raise IOError
                example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
                example_img = cv2.resize(example_img, self.output_shape)
                example_img = self.transform(example_img)

                return example_img, example_landmark, right_img,right_landmark

        elif self.train =='test':
            while True:
                image_path = self.test_data[index][0]
                landmark_path = self.test_data[index][1]
                landmark = np.load(landmark_path)

                right_landmark = landmark[self.test_data[index][2]]
                right_landmark = torch.FloatTensor((MS + np.dot(right_landmark.reshape(1,6), EIGVECS)).reshape(-1))
                print (right_landmark.shape)
                im = cv2.imread(image_path)
                if im is None:
                    raise IOError
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = cv2.resize(im, self.output_shape)
                im = self.transform(im)
                right_img = torch.FloatTensor(im)
                r = random.choice(
                    [x for x in range(1,30)])
                #load example image
                example_path = image_path[:-8] + '_%03d.jpg'%r
                example_landmark = landmark[self.train_data[r][2]]
                example_landmark = torch.FloatTensor((MS + np.dot(example_landmark.reshape(1,6), EIGVECS)).reshape(-1))

                example_img = cv2.imread(example_path)
                if example_img is None:
                    raise IOError
                example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
                example_img = cv2.resize(example_img, self.output_shape)
                example_img = self.transform(example_img)
                return example_img, example_landmark, right_img,right_landmark
        elif  self.train =='demo':
            landmarks = np.load('/home/lchen63/obama_fake.npy')
            landmarks =np.reshape(landmarks, (landmarks.shape[0], 136))
            while True:
                # try:

                    image_path = self.demo_data[index][0]
                    im = cv2.imread(image_path)
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    im = cv2.resize(im, self.output_shape)
                    im = self.transform(im)

                    right_img = torch.FloatTensor(im)
                    example_path = '/mnt/disk1/dat/lchen63/lrw/demo/musk1_region.jpg'

                    example_landmark = landmarks[0]


                    example_lip = cv2.imread(example_path)

                    example_lip = cv2.cvtColor(example_lip, cv2.COLOR_BGR2RGB)
                    example_lip = cv2.resize(example_lip, self.output_shape)
                    example_lip = self.transform(example_lip)


                    right_landmark = torch.FloatTensor(landmarks[index-1])

                    wrong_landmark = right_landmark
                    return example_lip, example_landmark, right_img,right_landmark, wrong_landmark


    def __len__(self):
        if self.train=='train':
            return len(self.train_data)
        elif self.train=='test':
            return len(self.test_data)
        else:
            return len(self.demo_data)

#############################################################grid
class GRIDdataset1D_single_gt(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 output_shape=[128, 128],
                 train='train'):
        self.train = train
        self.dataset_dir = dataset_dir
        self.output_shape = tuple(output_shape)

        if not len(output_shape) in [2, 3]:
            raise ValueError("[*] output_shape must be [H,W] or [C,H,W]")

        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "lmark_train.pkl"), "rb")
            self.train_data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "lmark_test.pkl"), "rb")
            self.test_data = pickle.load(_file)
            _file.close()
        elif self.train =='demo' :
            _file = open(os.path.join(dataset_dir, "new_img_full_gt_demo.pkl"), "rb")
            self.demo_data = pickle.load(_file)
            _file.close()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
        if self.train=='train':

                #load righ img
                image_path = os.path.join('/mnt/ssd0/dat/lchen63/grid/data' , self.train_data[index][0], self.train_data[index][0], '%05d.jpg'%(self.train_data[index][1] + 1))

                landmark_path = os.path.join('/mnt/ssd0/dat/lchen63/grid/data' , self.train_data[index][0], self.train_data[index][0] + '_norm_lmarks.npy')

                landmark = np.load(landmark_path)

                right_landmark = landmark[self.train_data[index][1]]

                right_landmark = torch.FloatTensor(right_landmark.reshape(-1))

                im = cv2.imread(image_path)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = cv2.resize(im, self.output_shape)
                im = self.transform(im)
                right_img = torch.FloatTensor(im)

                r = random.choice(
                    [x for x in range(1, 76)])
                example_path =   os.path.join('/mnt/ssd0/dat/lchen63/grid/data' , self.train_data[index][0], self.train_data[index][0], '%05d.jpg'%(r))
                example_landmark = landmark[r - 1]

                example_landmark = torch.FloatTensor(example_landmark.reshape(-1))

                example_img = cv2.imread(example_path)
                example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
                example_img = cv2.resize(example_img, self.output_shape)
                example_img = self.transform(example_img)
                return example_img, example_landmark, right_img,right_landmark




        elif self.train =='test':
            # try:
                #load righ img
            image_path = os.path.join('/mnt/ssd0/dat/lchen63/grid/data' , self.test_data[index][0], self.test_data[index][0], '%05d.jpg'%(self.test_data[index][1] + 1))

            landmark_path = os.path.join('/mnt/ssd0/dat/lchen63/grid/data' , self.test_data[index][0], self.test_data[index][0] + '_norm_lmarks.npy')

            landmark = np.load(landmark_path)

            right_landmark = landmark[self.test_data[index][1]]

            right_landmark = torch.FloatTensor(right_landmark.reshape(-1))

            im = cv2.imread(image_path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, self.output_shape)
            im = self.transform(im)
            right_img = torch.FloatTensor(im)

            r = random.choice(
                [x for x in range(1, 76)])
            example_path =   os.path.join('/mnt/ssd0/dat/lchen63/grid/data' , self.test_data[index][0], self.test_data[index][0], '%05d.jpg'%(r))
            example_landmark = landmark[r - 1]


            example_landmark = torch.FloatTensor(example_landmark.reshape(-1))

            example_img = cv2.imread(example_path)
            example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
            example_img = cv2.resize(example_img, self.output_shape)
            example_img = self.transform(example_img)
            return example_img, example_landmark, right_img,right_landmark

        elif  self.train =='demo':

            # try:
                #load righ img
            image_path = '/mnt/ssd0/dat/lchen63/lrw/demo/regions/' +  self.demo_data[index][0]
            landmark_path = '/mnt/ssd0/dat/lchen63/lrw/demo/landmark1d/' + self.demo_data[index][1].replace('obama_', 'obama_ge_')
            right_landmark = np.load(landmark_path)
            right_landmark = torch.FloatTensor(right_landmark.reshape(-1))
            # print ('=========================')
            # print ('real path: ' +  image_path)
            im = cv2.imread(image_path)
            if im is None:
                print (image_path)
                raise IOError
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, self.output_shape)
            im = self.transform(im)
            right_img = torch.FloatTensor(im)


            example_path =   '../image/musk1_region.jpg'
            example_landmark = np.load('../image/musk1.npy')
            # tp = ( np.dot(example_landmark.reshape(1,6), EIGVECS))[0,:].reshape(68,3)
            # tp = tp[:,:-1].reshape(-1)
            example_landmark = torch.FloatTensor(example_landmark.reshape(-1))

            example_img = cv2.imread(example_path)
            example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
            example_img = cv2.resize(example_img, self.output_shape)
            example_img = self.transform(example_img)
            # print (right_landmark.size())

            return example_img, example_landmark, right_img,right_landmark

    def __len__(self):
        if self.train=='train':
            return len(self.train_data)
        elif self.train=='test':
            return len(self.test_data)
        else:
            return len(self.demo_data)


class GRIDdataset1D_lstm_gt(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 output_shape=[128, 128],
                 train='train'):
        self.train = train
        self.dataset_dir = dataset_dir
        self.output_shape = tuple(output_shape)

        if not len(output_shape) in [2, 3]:
            raise ValueError("[*] output_shape must be [H,W] or [C,H,W]")

        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "lmark_16_train.pkl"), "rb")
            self.train_data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "lmark_16_test.pkl"), "rb")
            self.test_data = pickle.load(_file)
            _file.close()
        elif self.train =='demo' :
            _file = open(os.path.join(dataset_dir, "new_16_full_gt_demo.pkl"), "rb")
            self.demo_data = pickle.load(_file)
            _file.close()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
        if self.train=='train':

            #load righ img

            image_path_root = os.path.join('/mnt/ssd0/dat/lchen63/grid/data' , self.train_data[index][0], self.train_data[index][0])

            landmark_path = os.path.join('/mnt/ssd0/dat/lchen63/grid/data' , self.train_data[index][0], self.train_data[index][0] + '_norm_lmarks.npy')

            current_frame_id =self.train_data[index][1]
            right_img = torch.FloatTensor(16,3,self.output_shape[0],self.output_shape[1])
            for jj in range(16):
                this_frame = current_frame_id + jj
                image_path =  os.path.join(image_path_root, '%05d.jpg'%this_frame)
                im = cv2.imread(image_path)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = cv2.resize(im, self.output_shape)
                im = self.transform(im)
                right_img[jj,:,:,:] = torch.FloatTensor(im)



            landmark = np.load(landmark_path)

            right_landmark = landmark[self.train_data[index][1] - 1 : self.train_data[index][1] + 15  ]

            right_landmark = torch.FloatTensor(right_landmark.reshape(16,136))

            r = random.choice(
                [x for x in range(1,76)])
            example_path =   os.path.join(image_path_root, '%05d.jpg'%r)
            example_landmark = landmark[r - 1]

            example_landmark = torch.FloatTensor(example_landmark.reshape(-1))

            example_img = cv2.imread(example_path)
            example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
            example_img = cv2.resize(example_img, self.output_shape)
            example_img = self.transform(example_img)

            return example_img, example_landmark, right_img,right_landmark


        elif self.train =='test':
            image_path_root = os.path.join('/mnt/ssd0/dat/lchen63/grid/data' , self.test_data[index][0], self.test_data[index][0])

            landmark_path = os.path.join('/mnt/ssd0/dat/lchen63/grid/data' , self.test_data[index][0], self.test_data[index][0] + '_norm_lmarks.npy')

            current_frame_id =self.test_data[index][1]
            right_img = torch.FloatTensor(16,3,self.output_shape[0],self.output_shape[1])
            for jj in range(16):
                this_frame = current_frame_id + jj
                image_path =  os.path.join(image_path_root, '%05d.jpg'%this_frame)
                im = cv2.imread(image_path)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = cv2.resize(im, self.output_shape)
                im = self.transform(im)
                right_img[jj,:,:,:] = torch.FloatTensor(im)



            landmark = np.load(landmark_path)

            right_landmark = landmark[self.test_data[index][1] - 1 : self.test_data[index][1] + 15  ]

            right_landmark = torch.FloatTensor(right_landmark.reshape(16,136))

            r = random.choice(
                [x for x in range(1,76)])
            example_path =   os.path.join(image_path_root, '%05d.jpg'%r)
            example_landmark = landmark[r - 1]

            example_landmark = torch.FloatTensor(example_landmark.reshape(-1))

            example_img = cv2.imread(example_path)
            example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
            example_img = cv2.resize(example_img, self.output_shape)
            example_img = self.transform(example_img)

            return example_img, example_landmark, right_img,right_landmark


    def __len__(self):
        if self.train=='train':
            return len(self.train_data)
        elif self.train=='test':
            return len(self.test_data)
        else:
            return len(self.demo_data)
class GRID_1D_lstm_landmark_pca(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 train='train'):
        self.train = train
        self.num_frames = 16
        self.root_path = '/mnt/ssd0/dat/lchen63/grid/data'
        self.pca = torch.FloatTensor(np.load('landmark/basics/U_grid.npy')[:,:6] )
        self.mean = torch.FloatTensor(np.load('landmark/basics/mean_grid.npy'))
        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "lmark_train.pkl"), "rb")
            self.train_data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "lmark_test.pkl"), "rb")
            self.test_data = pickle.load(_file)
            _file.close()
        elif self.train =='demo' :
            _file = open(os.path.join(dataset_dir, "img_demo.pkl"), "rb")
            self.demo_data = pickle.load(_file)
            _file.close()


    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
        if self.train=='train':
            try:
                lmark_path = os.path.join(self.root_path , self.train_data[index][0] , self.train_data[index][0] + '_norm_lmarks.npy')
                mfcc_path = os.path.join(self.root_path,  self.train_data[index][0],  self.train_data[index][0] +'_mfcc.npy')

                lmark = np.load(lmark_path) * 5.0
                lmark = torch.FloatTensor(lmark)
                lmark = lmark - self.mean.expand_as(lmark)
                lmark = torch.mm(lmark,self.pca)

                mfcc = np.load(mfcc_path)

                r = random.choice(
                    [x for x in range(6,50)])
                example_landmark =lmark[r,:]
                example_mfcc = mfcc[(r -3) * 4 : (r + 4) * 4, 1 :]

                mfccs = []
                for ind in range(1,17):
                    t_mfcc =mfcc[(r + ind - 3)*4: (r + ind + 4)*4, 1:]
                    t_mfcc = torch.FloatTensor(t_mfcc)
                    mfccs.append(t_mfcc)
                mfccs = torch.stack(mfccs, dim = 0)
                landmark  =lmark[r+1 : r + 17,:]

                example_mfcc = torch.FloatTensor(example_mfcc)
                return example_landmark, example_mfcc, landmark, mfccs
            except:
                self.__getitem__(index + 1)
        if self.train=='test':
            lmark_path = os.path.join(self.root_path , self.test_data[index][0] , self.test_data[index][0] + '_norm_lmarks.npy')
            mfcc_path = os.path.join(self.root_path,  self.test_data[index][0],  self.test_data[index][0] +'_mfcc.npy')

            lmark = np.load(lmark_path) * 5.0
            lmark = torch.FloatTensor(lmark)
            lmark = lmark - self.mean.expand_as(lmark)
            lmark = torch.mm(lmark,self.pca)

            mfcc = np.load(mfcc_path)

            r = random.choice(
                [x for x in range(3,70)])
            example_landmark =lmark[r,:]
            example_mfcc = mfcc[(r -3) * 4 : (r + 4) * 4, 1 :]

            mfccs = []
            for ind in range(1,17):
                t_mfcc =mfcc[(r + ind - 3)*4: (r + ind + 4)*4, 1:]
                t_mfcc = torch.FloatTensor(t_mfcc)
                mfccs.append(t_mfcc)
            mfccs = torch.stack(mfccs, dim = 0)
            landmark  =lmark[r+1 : r + 17,:]

            example_mfcc = torch.FloatTensor(example_mfcc)

            return example_landmark, example_mfcc, landmark, mfccs

    def __len__(self):
        if self.train=='train':
            return len(self.train_data)
        elif self.train=='test':
            return len(self.test_data)
        else:
            pass


class GRID_1D_single_landmark_pca(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 train='train'):
        self.train = train
        self.num_frames = 16
        self.root_path = '/mnt/ssd0/dat/lchen63/grid/data'
        self.pca = torch.FloatTensor(np.load('landmark/basics/U_grid.npy')[:,:6] )
        self.mean = torch.FloatTensor(np.load('landmark/basics/mean_grid.npy'))
        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "lmark_train.pkl"), "rb")
            self.train_data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "lmark_test.pkl"), "rb")
            self.test_data = pickle.load(_file)
            _file.close()
        elif self.train =='demo' :
            _file = open(os.path.join(dataset_dir, "img_demo.pkl"), "rb")
            self.demo_data = pickle.load(_file)
            _file.close()

    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
        if self.train=='train':
            # try:

                lmark_path = os.path.join(self.root_path , self.train_data[index][0] , self.train_data[index][0] + '_norm_lmarks.npy')
                mfcc_path = os.path.join(self.root_path,  self.train_data[index][0],  self.train_data[index][0] +'_mfcc.npy')
                ind = self.train_data[index][1]

                lmark = np.load(lmark_path) * 5.0
                lmark = torch.FloatTensor(lmark)
                lmark = lmark - self.mean.expand_as(lmark)
                lmark = torch.mm(lmark,self.pca)


                mfcc = np.load(mfcc_path)

                r = random.choice(
                    [x for x in range(6,50)])
                example_landmark =lmark[r,:]
                t_mfcc =mfcc[(ind - 3)*4: (ind + 4)*4, 1:]

                t_mfcc = torch.FloatTensor(t_mfcc)
                landmark  =lmark[ind,:]

                return example_landmark, t_mfcc, landmark, t_mfcc

        if self.train=='test':
            lmark_path = os.path.join(self.root_path , self.test_data[index][0] , self.test_data[index][0] + '_norm_lmarks.npy')
            mfcc_path = os.path.join(self.root_path,  self.test_data[index][0],  self.test_data[index][0] +'_mfcc.npy')

            lmark = np.load(lmark_path) * 5.0
            lmark = torch.FloatTensor(lmark)
            lmark = lmark - self.mean.expand_as(lmark)
            lmark = torch.mm(lmark,self.pca)

            mfcc = np.load(mfcc_path)

            r = random.choice(
                [x for x in range(3,70)])
            example_landmark =lmark[r,:]
            example_mfcc = mfcc[(r -3) * 4 : (r + 4) * 4, 1 :]

            mfccs = []
            for ind in range(1,17):
                t_mfcc =mfcc[(r + ind - 3)*4: (r + ind + 4)*4, 1:]
                t_mfcc = torch.FloatTensor(t_mfcc)
                mfccs.append(t_mfcc)
            mfccs = torch.stack(mfccs, dim = 0)
            landmark  =lmark[r+1 : r + 17,:]

            example_mfcc = torch.FloatTensor(example_mfcc)

            return example_landmark, example_mfcc, landmark, mfccs

    def __len__(self):
        if self.train=='train':
            return len(self.train_data)
        elif self.train=='test':
            return len(self.test_data)
        else:
            pass
