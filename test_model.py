import numpy as np

import cv2
import librosa
import tensorflow as tf


class TestModel:
    """Class to load the saved model and test it.

    Parameters
    ----------
    audio_file : str
        The path to the audio file to be tested.
    weights_path : str
        The path to the saved weights.
    json_path : str
        The path to the json file containing the model architecture.

    Attributes
    ----------
    classes : list
        The labels.
    audio_file
    weights_path
    json_path

    """

    def __init__(self, audio_file, weights_path, json_path):
        self.audio_file = audio_file
        self.weights_path = weights_path
        self.json_path = json_path
        self.classes = ['siren', 'street_music', 'drilling', 'dog_bark',
                        'children_playing', 'gun_shot', 'engine_idling',
                        'air_conditioner', 'jackhammer', 'car_horn']

    def load_model(self):
        """Load the model.

        Returns
        -------
        model: tf.keras.models
            The loaded model.

        """
        json_file = open(self.json_path, 'r')
        loaded_json_model = json_file.read()
        model = tf.keras.models.model_from_json(loaded_json_model)
        model.load_weights(self.weights_path)
        return model

    def process_data(self):
        """Extract the spectrogram from the input audio file.

        Returns
        -------
        spectrogram: numpy ndarray
            The spectrogram of the input audio file.

        """
        audio_ts, sample_rate = librosa.load(self.audio_file, sr=None,
                                             res_type='kaiser_fast')
        S = librosa.feature.melspectrogram(y=audio_ts, sr=sample_rate)
        spectrogram = librosa.power_to_db(S, ref=np.max)
        spectrogram = cv2.resize(spectrogram, (64, 64))
        spectrogram = spectrogram / 255.0
        return spectrogram

    def run_test(self):
        """predict the category of the input audio file.

        Returns
        -------
        None.

        """
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.7
        tf.compat.v1.keras.backend.set_session(
            tf.compat.v1.Session(config=config))
        model = self.load_model()
        input_data = self.process_data()
        input_data = input_data.reshape(1, 64, 64, 3)
        prediction = model.predict_classes(input_data)
        print('Audio Class:', self.classes.index(prediction[0]))
