import os

import numpy as np

import cv2
import librosa
import librosa.display as dl
from matplotlib import pyplot as plt


def draw_spectrogram(audio_file):
    audio_ts, sample_rate = librosa.load(audio_file, sr=None, res_type='kaiser_fast')
    base_name = os.path.basename(audio_file)
    image_name = base_name.split('.')[0] + '.jpg'
    S = librosa.feature.melspectrogram(y=audio_ts, sr=sample_rate)
    spectrogram = librosa.power_to_db(S, ref=np.max)
    cv2.imwrite(os.path.join(destination_folder, image_name), spectrogram)