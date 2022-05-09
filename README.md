# DL4H-Project
 
This codebase is for CS598 Deep Learning for Healthcare Project by Carlos Alves Pereira and Wael Mobeirek that makes use of the paper Deep recurrent model for individualized prediction of alzheimer’s disease progression by Wonsik Jung, Eunji Jun and Heung-Il Suk

# Reference to Paper:

Wonsik Jung, Eunji Jun, Heung-Il Suk, Alzheimer’s Disease Neuroimaging Initiative, et al. 2021. Deep recurrent model for individualized prediction of alzheimer’s disease progression. NeuroImage, 237:118143.

# Links to Paper and Reference Code:
https://www.sciencedirect.com/science/article/pii/S1053811921004201?via%3Dihub 

Multiple parts of code reference original work at https://github.com/ssikjeong1/Deep_Recurrent_AD

# Dependencies

* Python 3.6+
* PyTorch 0.4.0+
* TensorFlow 1.3+ 

We created a requirements file for Jung's model to facilitate the process: 
1. Make sure that you have Python 3.9 by running the following in the command line: `python --version`
2. Clone/download the repository to your local machine
3. Using the command line, go to the main directory: `cd YOURPATH\DL4H-Project`
4. Create a python virual environment: `python -m venv venv`
5. Activate the virtual environment: `./venv\Scripts\activate`
6. Install the required depedencies: `pip install -r requirements.txt`

# Folders and descriptions of Preprocessing, Training and Evaluation Code

## Jung_base
This folder contains their original code base for their model editted to work with input files created by our code. Their input requires a `.pkl` file. Code to create this pkl file from the raw csv file is provided in this folder in `Zeros_Data_Processing_Jung.ipynb`.
### Running files in this folder
First file to be run is `Zeros_Data_Processing_Jung.ipynb`, which can be run through Juypter Notebooks. This will output a `.pkl` file that contains the data. To run the model, use the following commands
1. Using the command line, go to the main directory: `cd YOURPATH\DL4H-Project\Jung_base`
2. Activate the virtual environment: `./venv\Scripts\activate`
3. To run the model: `python main.py --data_path='pickle_file.pkl'`

## Jung_modified
This folder contains our additional experiment. It uses their base code with our additions of demographic and genetic features. Their input requires a pkl file. Code to create this pkl file from the raw csv file is provided in this folder in `Zeros_Data_Processing_Jung.ipynb`.
### Running files in this folder
First file to be run is `Zeros_Data_Processing_Jung.ipynb`, which can be run through Juypter Notebooks. This will output a `.pkl` file that contains the data. To run the model, use the following commands
1. Using the command line, go to the main directory: `cd YOURPATH\DL4H-Project\Jung_modified`
2. Activate the virtual environment: `./venv\Scripts\activate`
3. To run the model: `python main.py --data_path='pickle_file.pkl'`

## LSTM_baselines 
This folder contains two data processing files, which are the `.ipynb` files that take in a `.csv` file and create `.pkl` files. The `.py` files do the training and testing given the `.pkl` files created by the data processing files.
### Running files in this folder
Both .ipynb files can be run using jupyter notebooks. They output pkl files, which can be run by the .py files using the following commands:  
`python LSTM_cognitive_scores.py`  
`python LSTM_diagnosis.py`

# Table of Results
![alt text](https://github.com/calvespe/DL4H-Project/blob/master/Final%20Results.png)
