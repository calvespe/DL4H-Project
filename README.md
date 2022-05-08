# DL4H-Project
 
This codebase is for CS598 Deep Learning for Healthcare Project by Carlos Alves Pereira and Wael Mobeirek that makes use of the paper Deep recurrent model for individualized prediction of alzheimer’s disease progression by Wonsik Jung, Eunji Jun and Heung-Il Suk

# Reference to Paper:

Wonsik Jung, Eunji Jun, Heung-Il Suk, Alzheimer’s Disease Neuroimaging Initiative, et al. 2021. Deep recurrent model for individualized prediction of alzheimer’s disease progression. NeuroImage, 237:118143.

# Link to Paper:
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

## Jung_modified
This folder contains our additional experiment. It uses their base code with our additions of demographic and genetic features. Their input requires a pkl file. Code to create this pkl file from the raw cvs file is provided in this folder in Zeros_Data_Processing_Jung.ipynb.  

## LSTM_baselines 
This folder contains two data processing files, which are the .ipynb files that take in a csv file and create pkl files. The .py files do the training and testing given the pkl files created by the data processing files. 

