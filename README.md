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

All required python modules can be obtained through pip installs.

# Folders and descriptions

## Jung_base
This folder contains their original code base for their model editted to work with input files created by our code. Their input requires a pkl file. Code to create this pkl file from the raw csv file is provided in this folder in Zeros_Data_Processing_Jung.ipynb.
### Running files in this folder
First file to be run is Zeros_Data_Processing_Jung.ipynb, which can be run through Juypter Notebooks. This will output a pkl file, which will be used as input to the model in the following command:  
python main.py --dataset='Zero' --data_path=PATH --kfold=5 --impute_weigh=0.1 --reg_weight=0.5 --label_weight=0.5 --gamma=5.0 --cognitive_score=True  
Replace path with path to pkl when running

## Jung_modified
This folder contains our additional experiment. It uses their base code with our additions of demographic and genetic features. Their input requires a pkl file. Code to create this pkl file from the raw csv file is provided in this folder in Zeros_Data_Processing_Jung.ipynb.
### Running files in this folder
First file to be run is Zeros_Data_Processing_Jung.ipynb, which can be run through Juypter Notebooks. This will output a pkl file, which will be used as input to the model in the following command:  
python main.py --dataset='Zero' --data_path=PATH --kfold=5 --impute_weigh=0.1 --reg_weight=0.5 --label_weight=0.5 --gamma=5.0 --cognitive_score=True  
Replace path with path to pkl when running

## LSTM_baselines 
This folder contains two data processing files, which are the .ipynb files that take in a csv file and create pkl files. The .py files do the training and testing given the pkl files created by the data processing files.
### Running files in this folder
Both .ipynb files can be run using jupyter notebooks. They output pkl files, which can be run by the .py files using the following commands:  
python LSTM_cognitive_scores.py  
python LSTM_diagnosis.py
