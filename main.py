import argparse

from preporcessing import prepare_data
from test_model import TestModel
from train_model import TrainModel

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', required=True,
                    help='Run training mode or testing mode')
args = parser.parse_args()
mode = args.mode

if mode == 'train':
    audio_path = input('Please enter the audio files path: ')
    csv_file = input('Please enter the csv_file path: ')
    destination_folder = input(
        'Please enter the folder where the spectrograms will be saved: ')
    train_data = input(
        'Please enter the name of the npy file (train_data.npy): ')
    prepare_data(audio_path=audio_path,
                 destination_folder=destination_folder,
                 csv_file=csv_file,
                 npy_file=train_data)
    train = TrainModel(data_path=train_data)
    train.run_training()
elif mode == 'test':
    test_audio = input('Please enter the audio file path: ')
    weights_path = input('Please enter the weights path: ')
    json_file = input('Please enter the json file path: ')
    test = TestModel(audio_file=test_audio,
                     weights_path=weights_path,
                     json_path=json_file)
else:
    print('Please select between train or test modes')
