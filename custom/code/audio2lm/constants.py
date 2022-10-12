import torch
import os
from config import config


def create_dir(path):
    '''
    Create dir if not exist
    '''
    cwd = os.getcwd()
    assert os.path.basename(os.path.join(cwd,'fit3162_fit3164/custom')) == 'custom'
    full_path = os.path.join(cwd,  path)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return path


ACTOR = 'TESTACTOR'

EMOTION_NET_DATASET_DIR = create_dir(f'fit3162_fit3164/custom/data/mfcc/{ACTOR}/')
# EMOTION_NET_MODEL_DIR = create_dir(f'saved_models/EmotionNet/{ACTOR}/')
# EMOTION_NET_LOG_DIR = create_dir(f'log/EmotionNet_original/{ACTOR}/')


# AUTOENCODER_2X_DATASET_DIR = create_dir(f'data/emotion_length/{ACTOR}/') ## NOTE: generated from mfcc_dtw.py
# AUTOENCODER_2X_MODEL_DIR = create_dir(f'saved_models/AutoEncoder2x/{ACTOR}/')
# # AUTOENCODER_2X_RESUME_DIR = create_dir(f'/media/thea/Data/New_exp/3_intensity_{ACTOR}/SER_intensity_3/model/81_pretrain.pth')
# AUTOENCODER_2X_IMAGE_DIR = create_dir(f'image/{ACTOR}/')
# AUTOENCODER_2X_LOG_DIR = create_dir(f'log/AutoEncoder2x_pretrained_Ct_encoder/{ACTOR}/')


LM_ENCODER_DATASET_LANDMARK_DIR = create_dir(f'fit3162_fit3164/custom/data/landmark/landmark_68/{ACTOR}/generated_landmarks/') ## NOTE: generated from facial_landmark.py
LM_ENCODER_DATASET_MFCC_DIR = create_dir(f'fit3162_fit3164/custom/data/landmark/mfcc/{ACTOR}/') ## NOTE: generated from lm_pca.py
# LM_ENCODER_MODEL_DIR = create_dir(f'saved_models/ATEmotion/{ACTOR}/')


# LM_ENCODER_PRETRAINED_DIR = f'{AUTOENCODER_2X_MODEL_DIR}2_pretrain.pth'
LANDMARK_BASICS = create_dir(f'fit3162_fit3164/custom/data/landmark/landmark_68/{ACTOR}/basics/')

# ATPRETRAINED_DIR = f'pretrained_models/atnet_lstm_18.pth' ## NOTE: Ct_encoder pretrai
# SERPRETRAINED_DIR = f'{EMOTION_NET_MODEL_DIR}/SER_2.pkl'

# AUDIO_DATASET = create_dir(f'data/audio/{ACTOR}_wav/')
# ALIGNED_AUDIO_DATA = create_dir(f'data/aligned_audio/{ACTOR}/')
# EMOTION_LENGTH_OUTPUT = create_dir(f'data/emotion_length/{ACTOR}/')

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
