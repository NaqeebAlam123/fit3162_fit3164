import torch

ACTOR = 'M030'

EMOTION_NET_DATASET_DIR = f'data/mfcc/{ACTOR}'
EMOTION_NET_MODEL_DIR = f'saved_models_temp/EmotionNet/{ACTOR}/'
EMOTION_NET_LOG_DIR = f'log/EmotionNet_original/{ACTOR}/'

AUTOENCODER_2X_DATASET_DIR = f'data/emotion_length/{ACTOR}/' ## NOTE: generated from mfcc_dtw.py
AUTOENCODER_2X_MODEL_DIR = f'saved_models_temp/AutoEncoder2x/{ACTOR}/'
# AUTOENCODER_2X_RESUME_DIR = f'/media/thea/Data/New_exp/3_intensity_{ACTOR}/SER_intensity_3/model/81_pretrain.pth'
AUTOENCODER_2X_IMAGE_DIR = f'image/{ACTOR}/'
AUTOENCODER_2X_LOG_DIR = f'log/AutoEncoder2x_pretrained_Ct_encoder/{ACTOR}/'

LM_ENCODER_DATASET_LANDMARK_DIR = f'data/landmark/landmark_68/{ACTOR}/generated_landmarks/' ## NOTE: provided
LM_ENCODER_DATASET_MFCC_DIR = f'data/landmark/mfcc/{ACTOR}/' ## NOTE: generated from lm_pca.py
LM_ENCODER_MODEL_DIR = f'saved_models_temp/ATEmotion/{ACTOR}/'



LM_ENCODER_PRETRAINED_DIR = f'{AUTOENCODER_2X_MODEL_DIR}18_pretrain.pth'
LANDMARK_BASICS = f'data/landmark/landmark_68/{ACTOR}/basics/'

ATPRETRAINED_DIR = f'pretrained_models/atnet_lstm_18.pth' ## NOTE: Ct_encoder pretrain
SERPRETRAINED_DIR = f'{EMOTION_NET_MODEL_DIR}/SER_2.pkl'

AUDIO_DATASET = f'data/audio/{ACTOR}_wav/'

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
