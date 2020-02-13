#!/usr/bin/env python
# coding: utf-8

# # Last 4 channels of the input state 15X15.

# In[1]:


from torch.optim import SGD
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


# In[3]:


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


# In[11]:


def minmaxscale(v,min_v, max_v):
    return (v-min_v)/(max_v - min_v)

def recover_minmaxscale(s, min_v, max_v):
    return round(s*(max_v-min_v)+min_v)

def recover_xy(xy_normed, min_x = min_x, max_x = max_x, min_y = min_y, max_y = max_y):
    [x_normed, y_normed] = xy_normed
    x = recover_minmaxscale(x_normed, min_x, max_x)
    y = recover_minmaxscale(y_normed, min_y, max_y)
    return [x,y]


# In[12]:


def construct_s_ini_with_xy(xy):
    [x,y] = xy
    x_normed = minmaxscale(x, min_x, max_x)
    y_normed = minmaxscale(y, min_y, max_y)
    temp = [x_normed, y_normed, 0.0, 0.0]
    temp.extend([0.0 for a in range(15*15-25)])
    # the corresponding distances to POIs
    for place in train_airport:
        temp.append(minmaxscale((abs(x-train_airport[place][0][0])+abs(y-train_airport[place][0][1])),min_ta,max_ta)) 
    temp.extend([0.0 for a in range(15*15*4)]) #last 4 channels initilized to zeros
    temp = np.resize(temp, [5, 15, 15])
    s_ini = (torch.from_numpy(np.array([temp]))).float()
    return s_ini


# In[15]:


def correct_x(x_var, x_ini, mask = [], fix_xy = True):
    '''
    x_var: The generated(updated) Torch variable of the input x
    
    output: Correct the generated x according to background knowledge, 
            i.e., range from 0~1, zero entries
                and fix the x,y, distance to POIs.
    '''
    x_corrected = deepcopy(x_var.data.numpy())
    #correct the range to 0~1
    x_corrected[x_corrected > 1] = 1.0
    x_corrected[x_corrected < 0] = 0.0
    #set the zero entries
    x_corrected[0][0][0][4:] = 0.0 #first row
    x_corrected[0][0][1:13] = 0.0 #1~5 rows
    x_corrected[0][0][13][:9] = 0.0 #6th row
                       
    if fix_xy:#fix the x,y location and the corresponding distance to POIs.
        x_corrected[0][0][0][:2] = deepcopy(x_ini[0][0][0][:2]) #reset x,y to the x,y in the initial state
        #reset the distance to POIs to the initial values
        x_corrected[0][0][13][9:] = deepcopy(x_ini[0][0][13][9:]) 
        x_corrected[0][0][14] = deepcopy(x_ini[0][0][14])
    if len(mask) != 0:
        x_corrected[0][1:] = x_corrected[0][1:] * mask
    
    x_tens = (torch.from_numpy(x_corrected)).float()
    return x_tens

def l1_norm(x_var):
    '''
    x_var: The generated(updated) Torch variable of the input x
    
    output: the l1 norm for some entries of the input.
    '''
    x_array = deepcopy(x_var.data.numpy())
    x_abs = abs(x_array)
    return np.sum(x_abs[0][1:])

def norm_2_real(x_n, x_min, x_max):
    return round(x_min+(x_n*(x_max - x_min)))

def cat_2_vars(x_var0, x_var1):
    x_temp_var = torch.cat([x_var0, x_var1])
    x_array = np.array([x_temp_var.data.numpy()])
    x_tens = (torch.from_numpy(x)).float()
    x_var = Variable(x_tens, requires_grad=True)
    return x_var


# In[16]:


def generate_unreach_mask(loc = [24,30], side_length = 15):
    xy_unreach = pickle.load(open('./data/xy_unreach_1.pkl','rb'))
    half_l = side_length//2
    mask = []
    for x in range(loc[0] - half_l, loc[0] + half_l+1):
        for y in range(loc[1] - half_l, loc[1] + half_l+1):
            if [x,y] in xy_unreach:
                mask.append(0.0)
            else:
                mask.append(1.0)
    mask = np.array(mask)
    mask.resize(side_length, side_length)
    return mask


# In[17]:


def act_max_0(mean_st, target_action = 1, initial_learning_rate = 1e-2, max_iter = 2000, net = net, reg = 0,              correct = False, loss_thresh = 1e-5, loc_xy = [24,23], reg_weight = 0.01, unreach_masked = False,              sc_weight = 0.0, mse_weight = 0.0):
    '''
    Activation maximization.
    unreach_masked: indicate whether filtering out the unreachable grids or not.
    sc_weight: the weight of the regularization term of spatial correlation, \
                without spatial correlation of sc_weight == 0.0.
    '''
    
    x_ini = construct_s_ini_with_xy(loc_xy)
    x_tens = deepcopy(x_ini)
    x_var = Variable(x_tens, requires_grad=True)
    mean_st_tens = (torch.from_numpy(np.array([mean_st]))).float()
    mse = nn.MSELoss()
    net.eval()
    if unreach_masked:
        unreach_mask = generate_unreach_mask(loc = loc_xy, side_length= 15)
    else:
        unreach_mask = []
    train_loss = []
    last_loss = 0.0
    for i in range(max_iter):
        optimizer = torch.optim.Adam([x_var], lr=initial_learning_rate)
        output = net.forward(x_var.float())
        if reg == 1:
            l1_norm = torch.norm(x_var[0][1:], p = 1) #l1-norm
            loss_am = -output[0, target_action] + reg_weight*l1_norm
        elif reg == 2:
            l2_norm = torch.norm(x_var[0][1:], p = 2) #l2-norm
            loss_am = -output[0, target_action] + reg_weight*l2_norm
#             print('output:', -output[0, target_action].item(), 'l2:', l2_norm.item(), 'sc_norm:', sc_norm.item(), 'loss:', loss_am.item())
        else:
            loss_am = -output[0, target_action]
        if (mse_weight > 0.0):
            loss_am = loss_am + mse_weight * mse(x_var[0][1:], mean_st_tens[0][1:])
        if(i%1== 0):
            train_loss.append(loss_am)
        if abs(loss_am - last_loss) <= loss_thresh:
            print('Finish training:', i, loss_am)
            break
        last_loss = loss_am
        net.zero_grad()
        # Backward
#         loss_am*=1.0
        loss_am.backward()
        # Update image
        optimizer.step()
        if correct:
            x_tens = correct_x(x_var, x_ini, mask=unreach_mask)
            x_var = Variable(x_tens, requires_grad=True)
        
    if correct:
        output = net.forward(x_var.float())
#         print(output[0, target_action], loss_am)
        return x_var, output, train_loss, unreach_mask, float(output[0, target_action]), float(mse(x_var[0][1:], mean_st_tens[0][1:]))
    else:
        output = net.forward(x_var.float())
#         print(output[0, target_action], loss_am)
        return x_var, output, train_loss, unreach_mask, float(output[0, target_action]), float(mse(x_var[0][1:], mean_st_tens[0][1:]))


# In[26]:


def _max_real_st(target_loc, target_ac, expert_st, test_st):
    '''
    Find the real state in a target location which can maximize the output of the target action.
    '''
    all_st = expert_st+test_st
    real_st_pool = []
    for st in all_st:
        loc = recover_xy(list(st[0][0][:2]))
        if loc == target_loc:
            real_st_pool.append(st)
    print('pool size:', len(real_st_pool))
    max_real_st = 0
    max_prob = 0.

    for st in real_st_pool:
        prob = net.forward(torch.FloatTensor(torch.from_numpy(np.asarray([st])).float()))[0][target_ac]
        if prob > max_prob:
            max_prob = prob
            max_real_st = st.copy()
    return max_real_st, max_prob
def _real_st_mean_p(target_loc, target_ac, expert_st, test_st):
    '''
    Find the real state in a target location which can maximize the output of the target action.
    '''
    all_st = expert_st+test_st
    real_st_pool = []
    for st in all_st:
        loc = recover_xy(list(st[0][0][:2]))
        if loc == target_loc:
            real_st_pool.append(st)
    print('pool size:', len(real_st_pool))
    max_real_st = 0
    max_prob = 0.
    
    p_list = []
    for st in real_st_pool:
        prob = net.forward(torch.FloatTensor(torch.from_numpy(np.asarray([st])).float()))[0][target_ac]
        p_list.append(prob.item())
    return mean(p_list)
def _mean_real_st(target_loc, expert_st=expert_st, test_st=test_st):
    all_st = expert_st+test_st
    real_st_pool = []
    for st in all_st:
        loc = recover_xy(list(st[0][0][:2]))
        if loc == target_loc:
            real_st_pool.append(st)
    print('pool size:', len(real_st_pool))
    ### mean channel calculation
    ### Calculate the mean value on each grid in each channel, the dimension of the output is the same as the state.
    mean_st = real_st_pool[0]
    for st in real_st_pool[1:]:
        mean_st += st
    mean_st = mean_st / len(real_st_pool)
    return mean_st


# In[19]:


def reconstruct_st(st_mixed):
    st_1d = np.resize(st_mixed,81*5)
    st_real_1d = []
    for i in range(81*5-(81-25)):
        if i==4:
            for j in range(81-25):
                st_real_1d.append(0.0)
            st_real_1d.append(st_1d[i])
        else:
            st_real_1d.append(st_1d[i])
    st_real = np.resize(st_real_1d, [5,9,9])
    print(j)
    return st_real


# In[45]:


x_max_tens, output, train_loss, unreach_mask, max_prob, min_mse = act_max_0(target_action=act,
                             loc_xy=loc,
                             mean_st=mean_st,
                             initial_learning_rate=5e-3,
                             max_iter=1000,
                             reg=0,
                             correct=True,
                             loss_thresh=1e-9,
                             net = net,
                             reg_weight=0.05,
                             sc_weight = 0.0,
                             mse_weight = 1,
                             unreach_masked=True) #1: 0.008, 2: 0.01
max_prob, min_mse

