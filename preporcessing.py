import glob
import os

import numpy as np

import cv2
import librosa
import pandas as pd
from tqdm import tqdm


def draw_spectrogram(audio_file, destination_folder):
    """Draw the spectrogram of an input audio file and save it to an image.

    Parameters
    ----------
    audio_file : str
        the path to the input audio file.
    destination_folder : str
        Where the image will be saves`.

    Returns
    -------
    None.

    """
    audio_ts, sample_rate = librosa.load(audio_file, sr=None,
                                         res_type='kaiser_fast')
    base_name = os.path.basename(audio_file)
    image_name = base_name.split('.')[0] + '.jpg'
    S = librosa.feature.melspectrogram(y=audio_ts, sr=sample_rate)
    spectrogram = librosa.power_to_db(S, ref=np.max)
    cv2.imwrite(os.path.join(destination_folder, image_name), spectrogram)


def img_to_array(images_path, labels_file, img_size, npy_file):
    """Save the images and their labels to a numpy file.

    Parameters
    ----------
    images_path : str
        The path to the images.
    labels_file : str
        The path to the labels file.
    img_size : tuple
        The dimensions of the image to be resized.
    npy_file: str
        The path to the npy file containing the training data

    Returns
    -------
    type
        Description of returned object.

    """
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
    np.save(npy_file, data)


def prepare_data(audio_path, destination_folder, csv_file, npy_file):
    """Launch the preporcessing process.

    Parameters
    ----------
    audio_path : str
        The path to the audio files.
    destination_folder : str
        Where to save the images and the npy file.
    csv_file : str
        The path to the labels file.
    npy_file: str
        The path to the npy file containing the training data

    Returns
    -------
    None.

    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for audio in tqdm(glob.glob(audio_path + '/*.wav')):
        draw_spectrogram(audio_file=audio,
                         destination_folder=destination_folder)
    img_to_array(images_path=destination_folder,
                 labels_file=csv_file,
                 img_size=(64, 64),
                 npy_file=npy_file)
