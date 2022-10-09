import cv2
import os
from imutils import face_utils
import imutils
import dlib
#import multiprocessing
import numpy as np
from Exceptions_Classes import *
#from google.colab.patches import cv2_imshow

def extracting_frames(video_path,number_of_frames):
    """
    Video is loaded from video path and then segregated into frames.The supplied number of frames existing in starting of the video are returned in
    a list.
    Parameters
    ----------
    video_path : path of video that needs to be segregated into frames
    number_of_frames : number of frames that needed to be collected

    Returns
    -------
    array containing frames data
    """
    # If path to video path does not exist, raise a PathNotFoundError
    if  not os.path.exists(video_path):
        raise PathNotFoundError(video_path,"Incorrect Video Path")
    else:
        # if the file is not an mp4 file, then generate FileNotFoundError
        extension=os.path.splitext(os.path.split(video_path)[1])[1]
        if extension != ".mp4":
            raise FileNotFoundError

    # Noramllizing path
    video_path = os.path.normpath(video_path)

    # Initiate Video capturing on video
    stream=cv2.VideoCapture(video_path)

    # get total number of frames in video
    total_number_of_frames=int(stream.get(cv2.CAP_PROP_FRAME_COUNT))

    # set starting number of frame
    stream.set(1 ,0)
    start=0
    image_lst=[]
    end=number_of_frames

    # if supplied number of frames are more than frames existing in Video , raise InvalidNumberofFrames
    if end>total_number_of_frames:
        raise InvalidNumberofFrames

    # while last frame required is not reached
    while start<end:
        # read frame
        _,image=stream.read()
        # add frame to list
        if image is not None:
            image_lst = image_lst + [image]
        # move to next frame
        start=start+1
    stream.release()
    return image_lst

def generate_landmarks_frame(image,face_detector,predictor):
   """
   this function goal is to generate landmark points from given frame
   Parameters
   ----------
   image : frame
   face_detector : dlib face detector
   predictor : pre-trained model used for identifying landmark points

   Returns
   -------
   np array containing landmark points for both x and y coordinates

   """

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

       # adding numpy array coordinates

       landmarks=[]
       for (x,y) in shape:
           landmarks=landmarks+ [x,y]
   return np.array(landmarks)

def facial_landmark(video_path,number_of_frames):
    """
    This function creates np array containing landmark points for given number of frames using helper functions or external function that
    are defined in the file

    Parameters
    ----------
    video_path : path of video that needs to be segregated into frames
    number_of_frames : number of frames that needed to be collected

    Returns
    -------
    array containing landmarks data for given number of frames
    """
    # using dlib face detector
    face_detector=dlib.get_frontal_face_detector()
    # building a predictor using dlib that is provided with a pre-trained model for context
    predictor = dlib.shape_predictor(r"../../../shape_predictor_68_face_landmarks.dat")
    # creating an empty np array with (number_of_frames) x 136 as dimensions
    video_landmarks=np.empty((number_of_frames,136),dtype=int)
    # looping through all frames, predicting landmraks and collecting them in one np array
    for (i,image) in enumerate(extracting_frames(video_path,number_of_frames)):
        video_landmarks[i]=generate_landmarks_frame(image,face_detector,predictor)
    return video_landmarks