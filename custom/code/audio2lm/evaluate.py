from pickle import FRAME
import numpy as np
import torch
import torch.utils
from torch.autograd import Variable
import librosa
from models import AT_emotion
import cv2
from config import config
#import scipy.misc
#from tqdm import tqdm
#import torchvision.transforms as transforms

import python_speech_features
import random
from constants import LANDMARK_BASICS, LM_ENCODER_MODEL_DIR, AUDIO_DATASET

LANDMARK_POINTS = 68
FRAME_WIDTH, FRAME_HEIGHT = 256, 256
VIDEO_PATH = 'data/video/M030/'
FPS = 25


def get_mean():
    return np.load(f'{LANDMARK_BASICS}mean_68.npy')

def get_pca():
    return np.load(f'{LANDMARK_BASICS}U_68.npy')

class VideoWriter(object):
    def __init__(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.path = VIDEO_PATH
        self.out = cv2.VideoWriter(self.path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
    def write_frame(self, frame):
        self.out.write(frame)
    def end(self):
        self.out.release()

# mouth_video_writer = VideoWriter()


def draw_mouth():
    landmark = landmark.reshape(106*2,)
    # draw mouth from mouth landmarks, landmarks: mouth landmark points, format: x1, y1, x2, y2, ..., x20,
    heatmap = 255*np.ones((FRAME_WIDTH, FRAME_HEIGHT, 3), dtype=np.uint8)
    circle_color = (255, 0, 0)
    line_color = (0, 255, 0)
   
    def draw_line(start_idx, end_idx):
        for pts_idx in range(start_idx, end_idx):
            cv2.line(heatmap, (int(landmark[pts_idx * 2]), int(landmark[pts_idx * 2 + 1])),
                     (int(landmark[pts_idx * 2 + 2]), int(landmark[pts_idx * 2 + 3])), line_color, 3)
    draw_line(0, 32)     # face 
    # EYEBROW + MOUTH
    draw_line(33, 37)     
    draw_line(38, 42)     
    draw_line(64, 67)     
    draw_line(68, 71) 
    
    draw_line(84, 90)     # upper outer
    draw_line(96, 100)   # upper inner
    draw_line(100, 103)   # lower inner
    draw_line(90, 95)    # lower outer
        
    cv2.line(heatmap, (int(landmark[33 * 2]), int(landmark[33 * 2 + 1])),
             (int(landmark[64 * 2]), int(landmark[64 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[37 * 2]), int(landmark[37* 2 + 1])),
             (int(landmark[67 * 2]), int(landmark[67 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[38 * 2]), int(landmark[38 * 2 + 1])),
             (int(landmark[68 * 2]), int(landmark[68 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[42 * 2]), int(landmark[42* 2 + 1])),
             (int(landmark[71 * 2]), int(landmark[71 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[96 * 2]), int(landmark[96 * 2 + 1])),
             (int(landmark[103 * 2]), int(landmark[103 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[84 * 2]), int(landmark[84 * 2 + 1])),
             (int(landmark[95 * 2]), int(landmark[95 * 2 + 1])), thickness=3, color=line_color)
    #LEFT EYE
    draw_line(52, 53)   # lower inner
    draw_line(54, 56)    # lower outer
    
    cv2.line(heatmap, (int(landmark[53 * 2]), int(landmark[53 * 2 + 1])),
             (int(landmark[72 * 2]), int(landmark[72 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[72 * 2]), int(landmark[72* 2 + 1])),
             (int(landmark[54 * 2]), int(landmark[54 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[73 * 2]), int(landmark[73 * 2 + 1])),
             (int(landmark[56 * 2]), int(landmark[56 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[57 * 2]), int(landmark[57* 2 + 1])),
             (int(landmark[73 * 2]), int(landmark[73 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[52 * 2]), int(landmark[52 * 2 + 1])),
             (int(landmark[57 * 2]), int(landmark[57 * 2 + 1])), thickness=3, color=line_color)
    #RIGHT EYE
    draw_line(58, 59)   # lower inner
    draw_line(60, 62)    # lower outer
    
    cv2.line(heatmap, (int(landmark[59 * 2]), int(landmark[59 * 2 + 1])),
             (int(landmark[75 * 2]), int(landmark[75 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[75 * 2]), int(landmark[75* 2 + 1])),
             (int(landmark[60 * 2]), int(landmark[60 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[76 * 2]), int(landmark[76 * 2 + 1])),
             (int(landmark[62 * 2]), int(landmark[62 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[63 * 2]), int(landmark[63* 2 + 1])),
             (int(landmark[76 * 2]), int(landmark[76 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[58 * 2]), int(landmark[58 * 2 + 1])),
             (int(landmark[63 * 2]), int(landmark[63 * 2 + 1])), thickness=3, color=line_color)
    
    #NOSE
    draw_line(43, 46)   # lower inner
    draw_line(47, 51)    # lower outer
    
    cv2.line(heatmap, (int(landmark[78 * 2]), int(landmark[78 * 2 + 1])),
             (int(landmark[80 * 2]), int(landmark[80 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[80 * 2]), int(landmark[80* 2 + 1])),
             (int(landmark[82 * 2]), int(landmark[82 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[47 * 2]), int(landmark[47 * 2 + 1])),
             (int(landmark[82 * 2]), int(landmark[82 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[79 * 2]), int(landmark[79* 2 + 1])),
             (int(landmark[81 * 2]), int(landmark[81 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[81 * 2]), int(landmark[81 * 2 + 1])),
             (int(landmark[83 * 2]), int(landmark[83 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[51 * 2]), int(landmark[51 * 2 + 1])),
             (int(landmark[83 * 2]), int(landmark[83 * 2 + 1])), thickness=3, color=line_color)
    # draw keypoints
    for pts_idx in range(106):
        cv2.circle(heatmap, (int(landmark[pts_idx * 2]), int(landmark[pts_idx * 2 + 1])), radius=3, thickness=-1,
                   color=circle_color)
    return heatmap


def change_mouth(fake_lmark, clip):
    if len(fake_lmark) < len(clip):
        clip = clip[:len(fake_lmark)]
    index = 0
    s = 1
    for i in range(len(fake_lmark)):
        lmark = fake_lmark[i]
        if (lmark[102][1] - lmark[98][1]) < s:
            s = lmark[102][1] - lmark[98][1]
            index = i
    close_mouth = fake_lmark[index] ## NOTE: detect the contact point of two lips
    c = np.array(clip, dtype = float)
    for i in range(1, len(c)):
        if c[i] == 0:
            if c[i-1] == 1:
                fake_lmark[i] = 0.8*fake_lmark[i-1] + 0.2*close_mouth
                c[i] = 0.8
                fake_lmark[i+1] = 0.6*fake_lmark[i-1] + 0.4*close_mouth
                c[i+1] = 0.6
                fake_lmark[i+2] = 0.4*fake_lmark[i-1] + 0.6*close_mouth
                c[i+2] = 0.4
                fake_lmark[i+3] = 0.2*fake_lmark[i-1] + 0.8*close_mouth
                c[i+3] = 0.2
            elif ((i+1)< len(c)) and (c[i+1] == 1):
                fake_lmark[i] = 0.8*fake_lmark[i+1] + 0.2*close_mouth
                c[i] = 0.8
                fake_lmark[i-1] = 0.6*fake_lmark[i+1] + 0.4*close_mouth
                c[i-1] = 0.6
                fake_lmark[i-2] = 0.4*fake_lmark[i+1] + 0.6*close_mouth
                c[i-2] = 0.4
                fake_lmark[i-3] = 0.2*fake_lmark[i+1] + 0.8*close_mouth
                c[i-3] = 0.2
    for i in range(len(c)):
        if c[i] == 0:
            ratio = random.uniform(0.9,1)
            fake_lmark[i] = (1-ratio)*fake_lmark[i] + ratio*close_mouth
    return fake_lmark


def load_speech_and_extract_feature(audio_file):
    speech, sr = librosa.load(audio_file, sr=16000)
    # clip = check_volume(speech,sr)
    
    speech = np.insert(speech, 0, np.zeros(1920))
    speech = np.append(speech, np.zeros(1920))
    mfcc = python_speech_features.mfcc(speech,16000,winstep=0.01)
    return mfcc


def process_mfcc(mfcc):
    processed_mfcc = []
    ind = 3
    while ind <= int(mfcc.shape[0]/4) - 4:
        t_mfcc = mfcc[(ind - 3)*4: (ind + 4)*4, 1:]
        t_mfcc = torch.FloatTensor(t_mfcc).cuda()
        processed_mfcc.append(t_mfcc)
        ind += 1
    processed_mfcc = torch.stack(processed_mfcc, dim=0)
    return processed_mfcc

audio_file = f'{AUDIO_DATASET}neutral/001.wav' ## TODO
emo_audio_file = f'{AUDIO_DATASET}happy/001.wav' ## TODO

with torch.no_grad():
    # load model
    encoder = AT_emotion(config)
    encoder.load_state_dict(torch.load(f'{LM_ENCODER_MODEL_DIR}atnet_emotion_99.pth'))
    encoder.eval()

    # load landmark
    pca = torch.FloatTensor(get_pca()[:,:16]).cuda()
    mean = torch.FloatTensor(get_mean()).cuda()

    landmark = get_mean()
    landmark = landmark.reshape(LANDMARK_POINTS,2)  #150*2 
    landmark =  landmark.reshape((1,landmark.shape[0]* landmark.shape[1])) #1.300

    landmark = Variable(torch.FloatTensor(landmark.astype(float)) ).cuda()

    landmark  = landmark - mean.expand_as(landmark)
    landmark = torch.mm(landmark,  pca)

    # load audio mfcc
    mfcc = load_speech_and_extract_feature(audio_file) ## NOTE
    input_mfcc = process_mfcc(mfcc)

    # load emotion audio mfcc
    mfcc_emo = load_speech_and_extract_feature(emo_audio_file) ## NOTE
    emo_mfcc = process_mfcc(mfcc_emo)
    print(emo_mfcc.shape, input_mfcc.shape)
    # trim for equal size
    if(emo_mfcc.size(0) > input_mfcc.size(0)):
        emo_mfcc = emo_mfcc[:input_mfcc.size(0),:,:]
    if(emo_mfcc.size(0) < input_mfcc.size(0)):
        n = input_mfcc.size(0) - emo_mfcc.size(0)
        add = emo_mfcc[-1,:,:].unsqueeze(0)
        for i in range(n):
            emo_mfcc = torch.cat([emo_mfcc,add],0)
    print(emo_mfcc.shape, input_mfcc.shape)
    input_mfcc = input_mfcc.unsqueeze(0)
    emo_mfcc = emo_mfcc.unsqueeze(0)
    fake_lmark = encoder(landmark.cuda(), input_mfcc.cuda(), emo_mfcc.cuda())
    fake_lmark = fake_lmark.view(fake_lmark.size(0)*fake_lmark.size(1) , 16)

## TODO post process
fake_lmark = fake_lmark + landmark.expand_as(fake_lmark)
fake_lmark = torch.mm( fake_lmark, pca.t() )
fake_lmark = fake_lmark + mean.expand_as(fake_lmark)
fake_lmark = fake_lmark.unsqueeze(0)  
fake_lmark = fake_lmark.data.cpu().numpy()
fake_lmark = np.reshape(fake_lmark, (fake_lmark.shape[1], LANDMARK_POINTS,2))
torch.save(fake_lmark, 'predicted_lmark.pt')
# print(fake_lmark)

# clip = check_volume(speech,sr)
# fake_lmark = change_mouth(fake_lmark, clip)
# # np.save(config['sample_dir'], fake_lmark)

# mouth_img = []
# for i in range(len(fake_lmark)):
#     mouth_img.append(draw_mouth(fake_lmark[i]*255, 256, 256))
#     mouth_video_writer.write_frame(draw_mouth(fake_lmark[i]*255, 256, 256))
# mouth_video_writer.end()

# add_audio(config['video_dir'], opt.audio)

# print ('The generated video is: {}'.format(config['video_dir'].replace('.mp4','.mov')))

