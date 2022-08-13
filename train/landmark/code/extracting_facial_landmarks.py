import os
import numpy as np
from facial_landmarks import facial_landmark

def extracting_facial_landmarks(path, store_path):
    # set the directory to add npy_video folder for later
    os.chdir(os.path.abspath(os.path.join(path, os.pardir)))
    #print(os.getcwd())

    index = 0
    for dirname,_, filenames in os.walk(path):
        if index == 0:
            # make a new directory
            os.mkdir('landmark')
        os.chdir(store_path)
        
        npy_folder_label=dirname.split('\\')[-1]
        #print(npy_folder_label)
        
        if index>0:
            os.mkdir(npy_folder_label)
      
        for filename in filenames:
            npy_file_name=filename.split('.')[0]
            #print(npy_file_name)
            
            array=facial_landmark(os.path.join(dirname, filename))
            #print(os.path.join(dirname, filename))
            
            # switching directory to add the .npy file to the required directory
            os.chdir(store_path+ "\\" + str(npy_folder_label))
            #print(os.getcwd())
            np.save((store_path)+"\\"+str(npy_folder_label)+"\\"+str(npy_file_name)+".npy", array)
            
            # make a directory and save each of them individually as npy files (np.save)
            # split train and validation set and calculate U and mean (pca)
            #counter+=1
        if index > 0:
            print('Landmark npy files created for ' + str(npy_folder_label))    
        os.chdir(store_path)    
        index+=1
    print('Operation Completed')      



store_path = r'D:\Project1\fit3162_fit3164\dataset\train\train\landmark\dataset_M030\landmark'
path=r'D:\Project1\fit3162_fit3164\dataset\train\train\landmark\dataset_M030\video'
extracting_facial_landmarks(path, store_path)