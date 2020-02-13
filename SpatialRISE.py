#!/usr/bin/env python
# coding: utf-8

# In[1]:


## %pylab inline
import pickle
import pandas as pd
import csv
import re
import time
import os
import numpy as np
import itertools
from numpy import mean
from copy import deepcopy
from matplotlib import pyplot as plt
import matplotlib
from scipy import stats
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.autograd import Variable
import random
import scipy.optimize
import scipy.ndimage
import statistics
import collections


# In[2]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(5, 3, 3,padding=1)
        self.pool = nn.MaxPool2d(2,1)
        self.conv2 = nn.Conv2d(3, 2, 3)
        self.conv3 = nn.Conv2d(2, 2, 3)
#         self.conv4 = nn.Conv2d(2, 1, 2)
        
        self.fc1 = nn.Linear(162, 120) 
        self.fc2 = nn.Linear(120, 84) 
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 162)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.fc3(x)
        return F.softmax(x, dim=1)


    def select_action(self, x):
        action_prob = self.forward(x)
        action = action_prob.multinomial(1)
        return action
    def targeting_prob(self, x, labels):
        action_prob = self.forward(x)
        return action_prob.gather(1, labels)


# In[3]:


def generate_mask(zero_prob=0.3, h=13, H=19):
    mask_list = []
    for i in range(h*h):
        mask_list.append(float(np.random.choice([0,1],p = [zero_prob, 1-zero_prob])))
    mask_list = np.array(mask_list).reshape(h,h)
    mask = scipy.ndimage.zoom(mask_list, H/h, order=2)
    return mask

def generate_mask_5channels(zero_prob=0.3, h=7, sidelength = 15):
    H = sidelength
    C = H/h
    zoom_size = np.ceil((h+1)*C)
#     print(zoom_size)
    max_indent = zoom_size-sidelength
    mask = []
    for c in range(5):
        mask_list = []
        for i in range(h*h):
            mask_list.append(float(np.random.choice([0,1],p = [zero_prob, 1-zero_prob])))
        mask_list = np.array(mask_list).reshape(h,h)
    #     print(mask_list)
        mask_zoom = scipy.ndimage.zoom(mask_list, zoom_size/h, order=2)
        x_indent = random.randint(0, max_indent)
        y_indent = random.randint(0, max_indent)
        mask_crop = mask_zoom[x_indent:x_indent+sidelength, y_indent:y_indent+sidelength]
        mask.append(mask_crop)
    return np.array(mask)

def generate_mask_pixels(zero_prob = 0.3, sidelength = 15):
    H = sidelength
    mask = []
    for c in range(5):
        mask_list = []
        for i in range(H*H):
            mask_list.append(float(np.random.choice([0,1],p = [zero_prob, 1-zero_prob])))
        mask_list = np.array(mask_list).reshape(H,H)
        mask.append(mask_list)
    return np.array(mask)
    

def apply_mask(st, mask):
    temp_st = deepcopy(st)
    temp_st[1:] = np.multiply(temp_st[1:],mask[1:])
    return temp_st


# In[7]:


def generate_mask_cluster(labels_4chs, spatial_corr_4chs, zero_prob = 0.3):
#     zero_prob = 0.3
    H = 15
    mask = []#5 chnnels, the first channel is tribute
    for c in range(5):
        if c == 0:
            mask_list = [0.3] * H**2 #first tribute channel
        else:
            labels_ch = labels_4chs[c-1]
            spatial_corr = spatial_corr_4chs[c-1]
            seed_value = {}
            seed_index = {}
            mask_list = []

            ###Randomly generate seed value in the mask.
            for l in set(labels_ch):
                if str(l) not in seed_value.keys():
                    seed_value[str(l)] = float(np.random.choice([0,1],p = [zero_prob, 1-zero_prob]))
                    indices = [i for i, x in enumerate(labels_ch) if x == 1]
                    seed_index[str(l)] = np.random.choice(indices)

            for i in range(len(labels_ch)):
                l = labels_ch[i]
                if l == -1:#unclustered grid, randomly assign value independently.
                    v = float(np.random.choice([0,1],p = [zero_prob, 1-zero_prob]))
                else:#clustered grids, assign value according to the seed value and the spatial corealtion.
                    seed_ind = seed_index[str(l)]
                    seed_v = seed_value[str(l)]
                    corr = spatial_corr[D[i]][D[seed_ind]]
                    corr_map = (corr+1)/2.0
                    gap = min(corr_map, 1-corr_map)
                    if seed_v == 1:
                        v = corr_map + round(random.uniform(0.0,1.0), 3)*(1-corr_map)
                    else:
                        v = 1 - corr_map - round(random.uniform(0.0,1.0), 3)*(1-corr_map)
                mask_list.append(v)
        mask.append(np.array(mask_list).reshape(H,H))
    return np.array(mask)


# In[8]:


def generate_mask_cluster_no_seed(labels_4chs, zero_prob = 0.3, random_weight = 0.2):
    '''
    labels_4chs: the cluster label in each channel for each grid in the channel.
    zero_prob: the probability of covering grids. 0 means covering, 1 means retaining.
    random_weight: the weight for the randomness of the mask value. 
                    mask_value(i) = 0 + random_weight*random(0,1)
                or  mask_value(i) = 1 - random_weight*random(0,1)
    '''
#     zero_prob = 0.3
    H = 15
    mask = []#5 chnnels, the first channel is tribute
    for c in range(5):
        if c == 0:
            mask_list = [0.3] * H**2 #first tribute channel
        else:
            labels_ch = labels_4chs[c-1]
            seed_value = {}
            seed_index = {}
            mask_list = []

            ###Randomly generate seed value in the mask.
            for l in set(labels_ch):
                if str(l) not in seed_value.keys():
                    seed_value[str(l)] = float(np.random.choice([0,1],p = [zero_prob, 1-zero_prob]))
                    indices = [i for i, x in enumerate(labels_ch) if x == 1]
                    seed_index[str(l)] = np.random.choice(indices)

            for i in range(len(labels_ch)):
                l = labels_ch[i]
                if l == -1:#unclustered grid, randomly assign value independently.
                    trend = float(np.random.choice([0,1],p = [zero_prob, 1-zero_prob]))
                    if trend == 0:
                        v = 0 + random_weight*(round(random.uniform(0.0,1.0), 3))
                    else:#trend == 1
                        v = 1 - random_weight*(round(random.uniform(0.0,1.0), 3))
                else:#clustered grids, assign value according to the seed value and the spatial corealtion.
                    seed_ind = seed_index[str(l)]
                    seed_v = seed_value[str(l)]
                    if seed_v == 1: #here seed_v is the trend
                        v = 1 - random_weight*(round(random.uniform(0.0,1.0), 3))
                    else:
                        v = 0 + random_weight*(round(random.uniform(0.0,1.0), 3))
                mask_list.append(v)
        mask.append(np.array(mask_list).reshape(H,H))
    return np.array(mask)


# In[9]:


def generate_mask_cluster_no_random(labels_4chs, spatial_corr_4chs, zero_prob = 0.3):
#     zero_prob = 0.3
    H = 15
    mask = []#5 chnnels, the first channel is tribute
    for c in range(5):
        if c == 0:
            mask_list = [0.3] * H**2 #first tribute channel
        else:
            labels_ch = labels_4chs[c-1]
            spatial_corr = spatial_corr_4chs[c-1]
            seed_value = {}
            seed_index = {}
            mask_list = []

            ###Randomly generate seed value in the mask.
            for l in set(labels_ch):
                if str(l) not in seed_value.keys():
                    seed_value[str(l)] = float(np.random.choice([0,1],p = [zero_prob, 1-zero_prob]))
                    indices = [i for i, x in enumerate(labels_ch) if x == 1]
                    seed_index[str(l)] = np.random.choice(indices)

            for i in range(len(labels_ch)):
                l = labels_ch[i]
                if l == -1:#unclustered grid, randomly assign value independently.
                    v = float(np.random.choice([0,1],p = [zero_prob, 1-zero_prob]))
                else:#clustered grids, assign value according to the seed value and the spatial correlation.
                    seed_ind = seed_index[str(l)]
                    seed_v = seed_value[str(l)]
                    corr = spatial_corr[D[i]][D[seed_ind]]
                    corr_map = (corr+1)/2.0
                    gap = min(corr_map, 1-corr_map)
                    corr_inc = abs(corr)+0.6
                    corr_inc = corr_inc if corr_inc<1.0 else 1.0
                    if seed_v == 1:
#                         v = corr_map
                        v = corr_inc
                    else:
                        v = 1 - corr_inc
                mask_list.append(v)
        mask.append(np.array(mask_list).reshape(H,H))
    return np.array(mask)


# In[10]:


def generate_mask_cluster_solid(labels_4chs, zero_prob = 0.3):
#     zero_prob = 0.3
    H = 15
    mask = []#5 chnnels, the first channel is tribute
    for c in range(5):
        if c == 0:
            mask_list = [0.3] * H**2 #first tribute channel
        else:
            labels_ch = labels_4chs[c-1]
#             spatial_corr = spatial_corr_4chs[c-1]
            seed_value = {}
            seed_index = {}
        
            mask_list = []

            ###Randomly generate seed value in the mask.
            for l in set(labels_ch):
                if str(l) not in seed_value.keys():
                    seed_value[str(l)] = float(np.random.choice([0,1],p = [zero_prob, 1-zero_prob]))
                    indices = [i for i, x in enumerate(labels_ch) if x == 1]
                    seed_index[str(l)] = np.random.choice(indices)

            for i in range(len(labels_ch)):
                l = labels_ch[i]
                if l == -1:#unclustered grid, randomly assign value independently.
                    v = float(np.random.choice([0,1],p = [zero_prob, 1-zero_prob]))
                else:#clustered grids, assign value according to the seed value and the spatial corealtion.
                    seed_ind = seed_index[str(l)]
                    seed_v = seed_value[str(l)]
                    v = seed_v
                mask_list.append(v)
        mask.append(np.array(mask_list).reshape(H,H))
    return np.array(mask)


# In[1]:


#pixel mask
sidelength = 15
num_masks = 1000
ac_i = 1
out_list = []
sum_mask = np.zeros([5, sidelength, sidelength])
zero_mask = np.zeros([5, sidelength, sidelength])
importance_map = np.zeros([5, sidelength, sidelength])
po_list = []
for m_i in range(num_masks):
    if m_i%(num_masks//10) == 0:
        print(m_i/num_masks)
    mask = generate_mask_pixels(sidelength=sidelength)
    sum_mask += mask
    masked_st = apply_mask(max_st, mask)
#     masked_st = np.multiply(max_st,mask)
    masked_st_tens = torch.FloatTensor(torch.from_numpy(np.asarray([masked_st])).float())
    po = (net.forward(masked_st_tens))[0].tolist()
    po_list.append(po[ac_i])
    importance_map+=mask*po[ac_i]
importance_map = importance_map/sum_mask
print(mean(po_list))
for c_i in range(5):
    plt.figure()
    plt.pcolor(importance_map[c_i],color = 'grey', cmap='Blues')
    plt.colorbar()


# In[2]:


#cluster mask with clusters based on LISA Gi
# labels_4chs = pickle.load(open('./data/labels_4chs_LISA_Gi.pkl','rb')) #the cluster labels of the grids.
# labels_4chs = pickle.load(open('./data/labels_4chs_LISA_Ii.pkl','rb')) #the cluster labels of the grids.
labels_4chs = pickle.load(open('./data/labels_4chs_LISA_Gi_{}_{}_2020.pkl'.format(loc_str, str(target_act)),'rb')) #the cluster labels of the grids.

sidelength = 15
num_masks = 1000
ac_i = target_act
out_list = []
sum_mask = np.zeros([5, sidelength, sidelength])
zero_mask = np.zeros([5, sidelength, sidelength])
importance_map = np.zeros([5, sidelength, sidelength])
po_list = []
for m_i in range(num_masks):
    if m_i%(num_masks//10) == 0:
        print(m_i/num_masks)
#     mask = generate_mask_cluster(labels_4chs=labels_4chs, spatial_corr_4chs=spatial_corr_4chs)
    mask = generate_mask_cluster_no_seed(labels_4chs=labels_4chs)
    sum_mask += mask
    masked_st = apply_mask(max_st, mask)
#     masked_st = np.multiply(max_st,mask)
    masked_st_tens = torch.FloatTensor(torch.from_numpy(np.asarray([masked_st])).float())
    po = (net.forward(masked_st_tens))[0].tolist()
    po_list.append(po[ac_i])
    importance_map+=mask*po[ac_i]
importance_map = importance_map/sum_mask
print(mean(po_list))
for c_i in range(5):
    plt.figure()
    plt.pcolor(importance_map[c_i],color = 'grey', cmap='Blues')
    plt.colorbar()

