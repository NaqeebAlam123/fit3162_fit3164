EMOTION_NET_DATASET_DIR = 'data/mfcc/M030'
EMOTION_NET_MODEL_DIR = 'saved_models_temp/EmotionNet/M030/'
EMOTION_NET_LOG_DIR = 'log/EmotionNet_original/M030/'

AUTOENCODER_2X_DATASET_DIR = 'data/emotion_length/M030/' ## NOTE: generated from mfcc_dtw.py
AUTOENCODER_2X_MODEL_DIR = 'saved_models_temp/AutoEncoder2x/M030/'
# AUTOENCODER_2X_RESUME_DIR = '/media/thea/Data/New_exp/3_intensity_M030/SER_intensity_3/model/81_pretrain.pth'
AUTOENCODER_2X_IMAGE_DIR = 'image/M030/'
AUTOENCODER_2X_LOG_DIR = 'log/AutoEncoder2x_pretrained_Ct_encoder/M030/'

LM_ENCODER_DATASET_LANDMARK_DIR = 'data/landmark/landmark_68/M030/generated_landmarks/' ## NOTE: provided
LM_ENCODER_DATASET_MFCC_DIR = 'data/landmark/mfcc/M030/' ## NOTE: generated from lm_pca.py
LM_ENCODER_MODEL_DIR = 'saved_models_temp/ATEmotion/M030/'
LM_ENCODER_PRETRAINED_DIR = f'{AUTOENCODER_2X_MODEL_DIR}18_pretrain.pth'
LANDMARK_BASICS = 'data/landmark/landmark_68/M030/basics/'

ATPRETRAINED_DIR = 'pretrained_models/atnet_lstm_18.pth' ## NOTE: Ct_encoder pretrain
SERPRETRAINED_DIR = f'{EMOTION_NET_MODEL_DIR}/SER_2.pkl'

AUDIO_DATASET = 'data/audio/M030_wav/'
