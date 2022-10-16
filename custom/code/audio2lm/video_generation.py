from email.mime import audio
import os
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import cv2
import imutils
import moviepy.video.io.ImageSequenceClip
import torch
from moviepy.editor import *
import time
import math
import sys
from pydub import AudioSegment

# from fit3162_fit3164.custom.code.audio2lm.evaluate import evaluate
# from fit3162_fit3164.custom.code.audio2lm.constants import AUDIO_DATASET
# from fit3162_fit3164.custom.code.audio2lm.Exception_classes import *

from evaluate import evaluate
from constants import AUDIO_DATASET
from Exception_classes import *

def video_generation(audio_files):
    if len(audio_files)==0:
        raise FileNotFoundError

    if len(audio_files)>2:
        raise OutOfBoundNumFiles
    elif len(audio_files)<2:
        raise OutOfBoundNumFiles
    else:
        start_time = time.process_time()
        #print("start time": start_time)
        audio_file, emo_audio_file = audio_files[0], audio_files[1]
        evaluate(audio_file, emo_audio_file)
        predicted_lmark = torch.load('predicted_lmark.pt')

        #predicted_lmark.shape
        draw_predicted_landmark(predicted_lmark)
        video_compilation()
        video_audio_compilation(audio_file)
        end_time= time.process_time()
        print("elapsed time:" + str(end_time-start_time) + 'second')
  

def draw_predicted_landmark(landmarks):
     # store path for images
    # storepath=f'data/image_compilation/test/'

    storepath=f'fit3162_fit3164/custom/data/image_compilation/test/'
    if not os.path.exists(storepath):
        os.makedirs(storepath)
    

    # display of coords for series of frame 
    for i in range(landmarks.shape[0]):
        landmark= landmarks[i]
        # generating black canvas
        my_img_3 = np.zeros((512, 512, 3), dtype = np.uint8)
        my_img_3.fill(255)
        
        # resizing 
        my_img_3=imutils.resize(my_img_3, width=500)
        
        # display for each frame
        for j in range(landmarks.shape[1]):
            # print(type(landmarks[i,j,0]))
            cv2.circle(my_img_3, (int(landmarks[i,j,0]), int(landmarks[i,j,1])), 1, (0, 0, 255), -1)
        
      
        
        # write image to  the particulat location
        cv2.imwrite(storepath+'Image'+str(i)+".jpg", my_img_3)

        # display set of coords (landmarks) for a particular frame
        #cv2.imshow("Image"+str(i+1),my_img_3)

        
    print('done frames generating' )


def video_compilation():
    image_folder=f'fit3162_fit3164/custom/data/image_compilation/test/'
    fps=20

    image_files = [os.path.join(image_folder,img)
                for img in os.listdir(image_folder)
                if img.endswith(".jpg")]
                
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)

    store_video_path='fit3162_fit3164/custom/data/video_compilation'
    #store_video_path = f'data/video_compilation'
    if not os.path.exists(store_video_path):
        os.makedirs(store_video_path)

    clip.write_videofile(os.path.join(store_video_path, 'test.mp4'))

    print('done video generating')



def video_audio_compilation(audio_path):
    # video path
    video_path=f'fit3162_fit3164/custom/data/video_compilation/test.mp4'

    length=length_check(audio_path)

    print("length of audio", length)
   
   # loading video dsa gfg intro video
    clip = VideoFileClip(video_path)

    
    # getting only first 4 seconds
    clip = clip.subclip(0, length)

    # loading audio file
    audioclip = AudioFileClip(audio_path).subclip(0, length)

    # adding audio to the video clip
    videoclip = clip.set_audio(audioclip)

    video_audio_path = f'fit3162_fit3164/custom/data/data/video_with_audio_compilation'

    if not os.path.exists(video_audio_path):
        os.makedirs(video_audio_path)

    # saving video clip
    videoclip.write_videofile(os.path.join('assets','video_test_with_audio.mp4'))

    print('done video with audio generating')

#print()
#print(video_generation([audio_file, emo_audio_file]))
def length_check(audio_path):
    # code reference: https://www.codespeedy.com/find-the-duration-of-a-wav-file-in-python/


    #loading audio file form our system
    sound = AudioSegment.from_file(audio_path)


    #duration calculation function
    sound.duration_seconds == (len(sound) / 1000.0)
    #seconds to minutes conversion
    minutes_duartion = int(sound.duration_seconds // 60)
    seconds_duration = round((sound.duration_seconds % 60),3)
    return math.floor(seconds_duration)