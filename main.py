import argparse

from preporcessing import prepare_data
from test_model import TestModel
from train_model import TrainModel

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', required=True,
                    help='Run training mode or testing mode')
parser.add_argument('-a', '--audio', default='Train',
                    help='The path to the audio files')
parser.add_argument('-d', '--destination', default='spectrogram',
                    help='Where to save the spectrograms of the audio files')
parser.add_argument('-c', '--csv', default='train.csv',
                    help='The path to the CSV file')
parser.add_argument('-t', '--train_data', defaul='train_data.npy',
                    help='The path the npy file')
args = parser.parse_args()
mode = args.mode
audio_path = args.audio
destination_folder = args.destination
csv_file = args.csv
train_data = args.train_data

if mode == 'train':
    prepare_data(audio_path=audio_path,
                 destination_folder=destination_folder,
                 csv_file=csv_file)
    train = TrainModel(data_path=train_data)
    train.run_training()
elif mode == 'test':
    test_audio = input('Please enter the audio file path')
    weights_path = input('Please enter the weights path')
    json_file = input('Please enter the json file path')
    test = TestModel(audio_file=test_audio,
                     weights_path=weights_path,
                     json_path=json_file)
else:
    print('Please select between train or test modes')
