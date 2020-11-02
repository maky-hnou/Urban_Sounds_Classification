import os

import numpy as np

import cv2
import librosa
import pandas as pd
from tqdm import tqdm


def draw_spectrogram(audio_file, destination_folder=None):
    audio_ts, sample_rate = librosa.load(audio_file, sr=None,
                                         res_type='kaiser_fast')
    base_name = os.path.basename(audio_file)
    image_name = base_name.split('.')[0] + '.jpg'
    S = librosa.feature.melspectrogram(y=audio_ts, sr=sample_rate)
    spectrogram = librosa.power_to_db(S, ref=np.max)
    if destination_folder:
        cv2.imwrite(os.path.join(destination_folder, image_name), spectrogram)


def img_to_array(images_path, labels_file, img_size):
    df = pd.read_csv(labels_file)
    labels = pd.unique(df['Class']).tolist()
    data = []
    for index, row in tqdm(df.iterrows()):
        labels_null = 10 * [0]
        img_name = os.path.join(images_path, '{}.jpg'.format(row['ID']))
        img = cv2.imread(img_name)
        resized_img = cv2.resize(img, img_size)
        normalized_img = resized_img / 255.0
        label_idx = labels.index(row['Class'])
        labels_null[label_idx] = 1
        data.append(np.array([np.array(normalized_img), labels_null]))
    np.save('train_data.npy', data)
