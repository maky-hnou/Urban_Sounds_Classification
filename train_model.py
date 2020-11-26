import os

import numpy as np

import tensorflow as tf


class TrainModel:
    """Class to create and train the model.

    Parameters
    ----------
    data_path : str
        The path to the npy file.

    Attributes
    ----------
    data_x : numpy ndarray
        An array of the spectrograms of the audio files.
    data_y : numpy ndarray
        An array of the categories of the audio.
    data_path

    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.data_x, self.data_y = self.load_data()

    def load_data(self):
        """Load the spectrograms and labels from the npy file.

        Returns
        -------
        images: numpy ndarray
            An array of the spectrograms of the audio files.
        labels:numpy ndarray
            An array of the categories of the audio.

        """
        data = np.load(self.data_path, allow_pickle=True)
        images = np.array([i[0] for i in data])
        labels = np.array([i[1] for i in data])
        return images, labels

    def create_model(self):
        """Create the model.

        Returns
        -------
        model: tf.keras.models
            The created model.

        """
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(
            3, 3), activation='relu', input_shape=(64, 64, 3)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                         activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
                                         activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
                                         activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        # Compiling using adam and categorical crossentropy
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        print(model.summary())
        return model

    def run_training(self):
        """Launch the training then save the trained model.

        Returns
        -------
        None.

        """
        train_x = self.data_x[:5000]
        train_y = self.data_y[:5000]
        valid_x = self.data_x[5000:]
        valid_y = self.data_y[5000:]
        if not os.path.exists('models'):
            os.makedirs('models/')
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.7
        tf.compat.v1.keras.backend.set_session(
            tf.compat.v1.Session(config=config))
        model = self.create_model()
        model.fit(train_x,
                  train_y,
                  batch_size=32,
                  validation_data=(valid_x, valid_y),
                  steps_per_epoch=108,
                  validation_steps=32,
                  epochs=250)
        model.save('models/sound_classfier.h5', save_format='h5')
        model_json = model.to_json()
        with open('models/model.json', 'w') as json_file:
            json_file.write(model_json)
