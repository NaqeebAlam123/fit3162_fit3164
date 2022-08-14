import cv2
import os
from imutils import face_utils
import imutils
import dlib
#import multiprocessing
import numpy as np
#from google.colab.patches import cv2_imshow

def extracting_frames(video_path,number_of_frames):
    video_path = os.path.normpath(video_path)   
    
    video_dr,filename=os.path.split(video_path)
    stream=cv2.VideoCapture(video_path)
    #end=int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    stream.set(1 ,0)
    start=0
    image_lst=[]
    end=number_of_frames
    while start<end:
        _,image=stream.read()
        if image is None:
            print("Exit")
            break
        #save_path=os.path.join(saving_path,"{:010d}.jpg".format(start)) 
        #print(save_path)
        #if not os.path.exists(save_path):
        #    cv2.imwrite(save_path, image)
        image_lst=image_lst+[image]
        start=start+1
    stream.release()
    return image_lst

def generate_landmarks_frame(image,face_detector,predictor):
    
   # Loading image
   #image=cv2.imread(path)
   # Resizing the image and converting to gray scale to exclude any other faces in image
   image=imutils.resize(image,width=500)
   gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
   # Applying the face detector
   rects = face_detector(image, 1)
   for (i, rect) in enumerate(rects):
       # determine the facial landmarks for the face region, then
       # convert the facial landmark (x, y)-coordinates to a NumPy
       # # array
       shape = predictor(gray, rect)
       shape = face_utils.shape_to_np(shape)
       # convert dlib's rectangle to a OpenCV-style bounding box
       # # [i.e., (x, y, w, h)], then draw the face bounding box
       # #(x, y, w, h) = face_utils.rect_to_bb(rect)
       # #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
       # # show the face number
       # #cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
       # #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
       # # loop over the (x, y)-coordinates for the facial landmarks
       # # and draw them on the image
       # #for (x, y) in shape:
       # #cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
       landmarks=[]
       for (x,y) in shape:
           landmarks=landmarks+ [x,y]
   return np.array(landmarks)

def facial_landmark(video_path):
    face_detector=dlib.get_frontal_face_detector()
    
    predictor = dlib.shape_predictor(r"D:\Project2\fit3162_fit3164\shape_predictor_68_face_landmarks.dat")
    # Loading image
    # #image=cv2.imread(path)
    # # Resizing the image and converting to gray scale to exclude any other faces in image
    number_of_frames=25
    video_landmarks=np.empty((number_of_frames,136),dtype=int)
    for (i,image) in enumerate(extracting_frames(video_path,number_of_frames)):
        video_landmarks[i]=generate_landmarks_frame(image,face_detector,predictor)
    return video_landmarks

#landmark=facial_landmark(r'D:\Project2\dataset\train\train\landmark\dataset_M030\video\angry\028.mp4')
#print(landmark.shape)