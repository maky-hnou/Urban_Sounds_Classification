# Urban_Sounds_Classification

![Python version][python-version]
[![GitHub issues][issues-image]][issues-url]
[![GitHub forks][fork-image]][fork-url]
[![GitHub Stars][stars-image]][stars-url]
[![License][license-image]][license-url]

## About this repo:  
An Urban sound classifier built and trained using Tensorflow 2.  

## Content of the repo:  
The project has been organized as follows:  
- `model/`: the folder containing the trained model.  
- `requirements.txt`: a text file containing the needed packages to run the repo.  
- `main.py`: the script used to launch the training or testing.  
- `prepocessing.py`: the code used to extract the spectrogram from the audio files and save them to npy file.  
- `test_model.py`: the code used to load the saved model and classify the audio files.  
- ` train_model.py`: the code used to build the model and train it with the training data.  

## How to run:  
*N.B:* use Python 3.8  

**1. Clone the repo:**  
on your terminal, run `git clone https://github.com/maky-hnou/Urban_Sounds_Classification.git`  
Then get into the project folder: `cd Urban_Sounds_Classification/`  
We need to install some dependencies:  
`sudo apt install python3-pip libpq-dev python3-dev`  

**2. Install requirements:**  
Before running the app, we need to install some packages.  
- *<ins>Optional</ins>* Create a virtual environment:  To do things in a clean way, let's create a virtual environment to keep things isolated.  
Install the virtual environment wrapper: `pip3 install virtualenvwrapper`  
Add the following lines to `~/.bashrc`:  
```
export WORKON_HOME=$HOME/.virtualenvs
export PROJECT_HOME=$HOME
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
export VIRTUALENVWRAPPER_VIRTUALENV=~/.local/bin/virtualenv
source ~/.local/bin/virtualenvwrapper.sh
```
Run `source ~/.bashrc`  
Run `mkvirtualenv sound_classfier`  
Activate the virtual environment: `workon sound_classfier` (To deactivate the virtual environment, run `deactivate`)  
- Install requirements: To install the packages needed to run the application, run `pip3 install -r requirements.txt`  

*N.B:* If you don't have GPU, or don't have Cuda and Cudnn installed, replace `tensorflow-gpu` by `tensorflow` in requirements.txt.  

**3- Download the dataset:**  
The dataset is available on [Google Drive](https://drive.google.com/drive/folders/0By0bAi7hOBAFUHVXd1JCN3MwTEU).  
Download the train and test files then extract them in the repository.

**4- Run the training or testing:**  
To run the training or the test the pre-trained model, run:  
```
python3 main.py --mode <train/test>
```
Then follow the steps; you'll be asked to give the path to files and folders based on the chosen mode.

[python-version]:https://img.shields.io/badge/python-3.8-brightgreen.svg
[issues-image]:https://img.shields.io/github/issues/maky-hnou/Urban_Sounds_Classification.svg
[issues-url]:https://github.com/maky-hnou/Urban_Sounds_Classification/issues
[fork-image]:https://img.shields.io/github/forks/maky-hnou/Urban_Sounds_Classification.svg
[fork-url]:https://github.com/maky-hnou/Urban_Sounds_Classification/network/members
[stars-image]:https://img.shields.io/github/stars/maky-hnou/Urban_Sounds_Classification.svg
[stars-url]:https://github.com/maky-hnou/Urban_Sounds_Classification/stargazers
[license-image]:https://img.shields.io/github/license/maky-hnou/Urban_Sounds_Classification.svg
[license-url]:https://github.com/maky-hnou/Urban_Sounds_Classification/blob/master/LICENSE
