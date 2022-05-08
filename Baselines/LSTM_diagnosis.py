import numpy as np
import pandas as pd
import scipy
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from scipy.stats import norm
import pickle
import csv
from sympy import Symbol, solve

Dataset = np.load('ADNI_DATA_JUNG_FORMAT_11_ave_test_masked.pkl', allow_pickle=True)

train_data = Dataset['Train_data']
train_label = Dataset['Train_label']

valid_data = Dataset['Valid_data']
valid_label = Dataset['Valid_label']

test_data = Dataset['Test_data']
test_label = Dataset['Test_label']
mask_test = Dataset['Mask_label']

train_encoding = Dataset['Train_Encoding']
validate_encoding = Dataset['Valid_Encoding']
test_encoding = Dataset['Test_Encoding']
print(train_data.shape)
print(train_label.shape)
print(mask_test.shape)
print(train_data[0])
print(train_label[0])
print(train_encoding.shape)

#normalize data
def normalize_feature(train_data):
    sequence_length = train_data.shape[1]
    tmp = []
    train_feature = train_data[:, :, 8:14]
    ICV_bl = train_data[:, :, 14]
    len = np.shape(train_feature)[-1]
    mask = np.ones_like(train_feature.reshape(-1,6))
    mask[np.where(train_feature.reshape(-1, 6) == 0)] = 0
    for idx in range(len):
        data = train_feature[:,:,idx]
        norm_data = np.true_divide(data, ICV_bl)
        tmp.append(norm_data)
        t_tmp = np.array(tmp).transpose(1, 2, 0)
    """ print('===debug normalize t_tmp', t_tmp.astype(float)) 
    print('missing bl: ', np.count_nonzero(~np.isnan(train_feature[:,0,:])))
    print(train_feature.shape) """
    return t_tmp.astype(float), mask.reshape(-1,sequence_length ,6).astype(float)

def masking_cogntive_score(data):
    sequence_length = data.shape[1]
    tmp = []
    max_range = [30,70,85]
    cog_feature = data.copy()
    mask = np.ones_like(cog_feature.reshape(-1,3))
    mask[np.where(cog_feature.reshape(-1,3)==0)] = 0
    for i in range(cog_feature.shape[2]):
        cog_data = cog_feature[:,:,i]
        norm_data = cog_data / max_range[i]
        tmp.append(norm_data)
        t_tmp = np.array(tmp).transpose(1,2,0)
    return t_tmp.astype(float), mask.reshape(-1, sequence_length, 3).astype(int)

def scaling_feature_e(train_feature, estim_m_out=None, estim_c_out=None, train=False):
    (b, s, f) = train_feature.shape
    tmp = train_feature.reshape(b*s, f)
    norm_train_feature = []
    norm_estim_c = []
    norm_estim_m = []

    for idx in range(tmp.shape[1]):
        tmp_vol = tmp[:, idx]
        if train == True:
            tmp_vol_max = np.max(tmp[:, idx])
            tmp_vol_min = np.min(tmp[np.nonzero(tmp[:, idx]), idx])

            m = Symbol('m')
            c = Symbol('c')
            equation1 = m * tmp_vol_max + c - 1
            equation2 = m * tmp_vol_min + c + 1
            """ print('===debug tmp.shape', tmp.shape)
            print('===debug tmp[:, idx]', tmp[:, idx])
            print('===debug tmp[:, idx].shape', tmp[:, idx].shape)
            print('===debug tmp_vol_max', tmp_vol_max)
            print('===debug tmp_vol_min', tmp_vol_min)
            print('===debug solve output', solve((equation1, equation2), dict=True))
            print('===debug m output', m) """
            estim_m = solve((equation1, equation2), dict=True)[0][m]
            estim_c = solve((equation1, equation2), dict=True)[0][c]
        else:
            estim_m = estim_m_out[idx]
            estim_c = estim_c_out[idx]
        norm_tmp_vol = (estim_m * tmp_vol) + estim_c
        norm_train_feature.append(norm_tmp_vol)
        norm_estim_m.append(estim_m)
        norm_estim_c.append(estim_c)
    norm_train_feature = np.array(norm_train_feature)
    norm_estim_m = np.array(norm_estim_m)
    norm_estim_c = np.array(norm_estim_c)

    norm_train_feature_t = norm_train_feature.transpose(1, 0).reshape(b, s, f)
    return norm_train_feature_t.astype(float), norm_estim_m.astype(float), norm_estim_c.astype(float)

# normalize to ICV
# Ventricles, Hippocampus, WholeBrain, Entorhinal, Fusiform, MidTemp
train_feature, train_mask = normalize_feature(train_data)
valid_feature, valid_mask = normalize_feature(valid_data)
test_feature, test_mask = normalize_feature(test_data)

norm_train_feature, estim_m, estim_c = scaling_feature_e(train_feature, None, None, train=True)
norm_valid_feature, v_estim_m, v_estim_c = scaling_feature_e(valid_feature, estim_m, estim_c, train=False)
norm_test_feature, t_estim_m, t_estim_c = scaling_feature_e(test_feature, estim_m, estim_c, train=False)

# Cognitive Score Case
mmse_train_feature = train_data[:, :, 3:6]
mmse_valid_feature = valid_data[:, :, 3:6]
mmse_test_feature = test_data[:, :, 3:6]

# Cognitive For Labels
mmse_train_labels = train_data[:, :, 3:6]
mmse_valid_labels = valid_data[:, :, 3:6]
mmse_test_labels = test_data[:, :, 3:6]

train_cog_norm_feature, train_cog_norm_mask = masking_cogntive_score(mmse_train_feature)
valid_cog_norm_feature, valid_cog_norm_mask = masking_cogntive_score(mmse_valid_feature)
test_cog_norm_feature, test_cog_norm_mask = masking_cogntive_score(mmse_test_feature)

model_train_input = np.concatenate((norm_train_feature, train_cog_norm_feature), axis=2)
model_train_mask = np.concatenate((train_mask, train_cog_norm_mask), axis=2)
model_valid_input = np.concatenate((norm_valid_feature, valid_cog_norm_feature), axis=2)
model_valid_mask = np.concatenate((valid_mask, valid_cog_norm_mask), axis=2)
model_test_input = np.concatenate((norm_test_feature, test_cog_norm_feature), axis=2)
model_test_mask = np.concatenate((test_mask, test_cog_norm_mask), axis=2)

model_train_input = np.delete(model_train_input, (-1), axis=1)
model_train_mask = np.delete(model_train_mask, (-1), axis=1)
model_valid_input = np.delete(model_valid_input, (-1), axis=1)
model_valid_mask = np.delete(model_valid_mask, (-1), axis=1)
model_test_input = np.delete(model_test_input, (-1), axis=1)
model_test_mask = np.delete(model_test_mask, (-1), axis=1)

# Get diagnosis
mmse_train_lables = train_encoding
mmse_valid_labels = validate_encoding
mmse_test_labels = test_encoding

model_train_labels = np.delete(mmse_train_labels, (0), axis=1)
model_valid_labels = np.delete(mmse_valid_labels, (0), axis=1)
model_test_labels = np.delete(mmse_test_labels, (0), axis=1)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# LSTM 2
class LSTMTagger(nn.Module):

    def __init__(self):
        super(LSTMTagger, self).__init__()
        
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(9, 64, 10, dropout=0.25)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2cog = nn.Linear(64, 3)
        self.hidden2diag = nn.Linear(64, 3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dp = nn.Dropout(p=0.5)
        

    def forward(self, x):
        h_n, c_n = self.lstm(x)
        hid = self.hidden2cog(h_n)
        cog_scores = self.relu(hid)
        diag_scores = self.softmax(hid)
        return diag_scores

model = LSTMTagger()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10):  # again, normally you would NOT do 300 epochs, it is toy data
    train_loss = 0
    for x, y in zip(model_train_input, model_train_labels):
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 3. Run our forward pass.
        y_pred = model(x)

        # Step 4. Compute the loss, gradients, and update the parameters by
        loss = loss_function(y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss / len(model_train_input)
    print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch+1, train_loss))

predictions = []
for x, y in zip(model_test_input, model_test_labels):
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)
    y_pred = model(x)
    y_pred = y_pred.detach().numpy()
    y = y.detach().numpy()
    y_pred = np.ndarray.tolist(y_pred)
    predictions.append(y_pred)
predictions = np.array(predictions)

pred_diag = predictions
real_diag = model_test_labels
pred_score = predictions
pred_score = pred_score.reshape(-1,3)

from sklearn.metrics import precision_score, precision_recall_fscore_support
from sklearn.metrics import recall_score, roc_auc_score
print(pred_diag.shape)
mask_test_final = np.delete(mask_test, [0], axis=1)
mask_test_final = mask_test_final.reshape(-1, 1)
pred_diag2 = pred_diag.argmax(2).reshape(-1, 1)
real_diag2 = real_diag.argmax(2).reshape(-1, 1)
pred_diag2 = pred_diag2[np.where(mask_test_final == 1)]
real_diag2 = real_diag2[np.where(mask_test_final == 1)]
pred_score = pred_score[np.where(mask_test_final == 1)]
real_diag = real_diag.reshape(-1, 3)
real_diag = real_diag[np.where(mask_test_final == 1)]
p, r, _, _ = precision_recall_fscore_support(real_diag2, pred_diag2, average='micro')
auc = roc_auc_score(real_diag, pred_score, average='macro', multi_class='ovr')
auc = roc_auc_score(real_diag, pred_score)
print('Precision')
print(p)
print('Recall')
print(r)
print('AUC score')
print(auc)