import os
import numpy as np
from facial_landmarks import facial_landmark
from pathlib import Path

def extracting_facial_landmarks(path):
    p=Path(path)
    for path in p.iterdir():
        print(path.name)
        if path.name == 'neutral':
            storepath = r'D:\Project2\fit3162_fit3164\train\landmark\dataset_M030\landmark\M030_'+path.name+'_1_'
        else:
            storepath = r'D:\Project2\fit3162_fit3164\train\landmark\dataset_M030\landmark\M030_'+path.name+'_3_'
        facial_landmark_all = []
        #os.makedirs(storepath,exist_ok=True)
        for i in range(len(list(path.iterdir()))):
            #type(str(path.joinpath(str(str(i).rjust(3, '0'))+'.mp4')))

           #facial_landmark(str(path.joinpath(str(str(i+1).rjust(3, '0'))+'.mp4'))))
           #D:\Project2\dataset\train\train\landmark\dataset_M030\video\angry
           #print(path.joinpath(str(str(i+1).rjust(3, '0'))+'.mp4'))
           #array=facial_landmark(str(path.joinpath(str(str(i+1).rjust(3, '0'))+'.mp4')))
           #print(storepath + str(str(i+1).rjust(3, '0')) + '\\' + '0.npy')
            array=[]
            os.makedirs(storepath+ str(str(i+1).rjust(3, '0')),exist_ok=True)
            np.save(storepath + str(str(i+1).rjust(3, '0')) + '\\' + '0.npy', array)
           #print(i+1)
        #print(np.array(facial_landmark_all).shape)
        #print((len(list(path.iterdir()))-4)//25)
        #print(facial_landmark_all.shape)
        break
    # # set the directory to add npy_video folder for later
    # os.chdir(os.path.abspath(os.path.join(path, os.pardir)))
    # #print(os.getcwd())

    # index = 0
    # for dirname,_, filenames in os.walk(path):
    #     if index == 0:
    #         # make a new directory
    #         os.mkdir('landmark')
    #     os.chdir(store_path)
        
    #     npy_folder_label=dirname.split('\\')[-1]
    #     #print(npy_folder_label)
        
    #     if index>0:
    #         os.mkdir(npy_folder_label)
      
    #     for filename in filenames:
    #         npy_file_name=filename.split('.')[0]
    #         #print(npy_file_name)
            
    #         array=facial_landmark(os.path.join(dirname, filename))
    #         #print(os.path.join(dirname, filename))
            
    #         # switching directory to add the .npy file to the required directory
    #         os.chdir(store_path+ "\\" + str(npy_folder_label))
    #         #print(os.getcwd())
    #         np.save((store_path)+"\\"+str(npy_folder_label)+"\\"+str(npy_file_name)+".npy", array)
            
    #         # make a directory and save each of them individually as npy files (np.save)
    #         # split train and validation set and calculate U and mean (pca)
    #         #counter+=1
    #     if index > 0:
    #         print('Landmark npy files created for ' + str(npy_folder_label))    
    #     os.chdir(store_path)    
    #     index+=1
    # print('Operation Completed')      



path=r'D:\Project2\dataset\train\train\landmark\dataset_M030\video'
extracting_facial_landmarks(path)