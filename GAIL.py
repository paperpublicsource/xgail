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


# In[12]:


def random_sample_inputs(states, length):
    output = random.sample(states, length)
    return torch.from_numpy(np.asarray(output))


# In[13]:


classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


# In[14]:


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


# In[15]:


class Dis(nn.Module):
    def __init__(self):
        super(Dis, self).__init__()
        self.label_embedding = nn.Embedding(10, 10)
        self.linear = nn.Linear(15*15*5+10, 15*15*5)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(5, 5, 3)
        self.pool = nn.MaxPool2d(2, 1)
        self.conv2 = nn.Conv2d(5, 3, 3)
        self.conv3 = nn.Conv2d(3, 1, 3)
        self.fc1 = nn.Linear(49, 36)
        self.fc2 = nn.Linear(36, 18)
        self.fc3 = nn.Linear(18, 1) 
    def forward(self, state, label):
        label = label.view(label.size(0))
        x = torch.cat((state.view(state.size(0), -1), self.label_embedding(label)), dim=1)
        x = self.relu(self.linear(x))
        x = x.view(x.size(0), 5, 15, 15)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 49)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.fc3(x)
        return torch.sigmoid(x)


# In[16]:


def cross_entropy(target, ground_truth):
    epsilon = 1e-12
    ce = 0.
    ce2 = 0.
    target = target.copy()
    ground_truth = ground_truth.copy()
    ces = []
    ce2s = []
    for state in range(len(ground_truth)):
        target_prime = np.clip(target[state], epsilon, 1.-epsilon)
        ground_truth[state] = np.clip(ground_truth[state], epsilon, 1.-epsilon)
        t = np.sum(ground_truth[state]*np.log((target_prime/ground_truth[state])))
        t2 = np.sum(ground_truth[state]*np.log(target_prime))
        ce2 -= t2
        ce -= t
        ces.append(t)
        ce2s.append(t2)
    return ce/(len(target)), ce2/(len(target)), ces

def difference(target, ground_truth):
    out = []
    out2 = []
    for i in range(len(ground_truth)):
        t = sum(abs(ground_truth[i] - target[i]))/10
        t2 = sum(abs(ground_truth[i] - target[i])**2)/10
        out.append(t)
        out2.append(t2)
    print(sum(out)/len(ground_truth), sum(out2)/len(ground_truth))
    return out, out2
def recover_minmaxscale(s, min_v, max_v):
    return round(s*(max_v-min_v)+min_v)


# In[ ]:


dtype = torch.float32
torch.set_default_dtype(dtype)
temp_kl = 0.45
temp_dis = 0
temp_net = 0
for epoch in range(200):
    if epoch%2 == 0:
        print(epoch+1)
    running_loss = 0.
    dis_loss = 0.
    for i, data in enumerate(trainloader, 0):
        
        '''expert inputs and labels(actions)'''
        inputs, labels = data
        inputs = inputs.float()
        labels = labels.long()
        
        batch_size = inputs.size(0)
        gen_inputs = (random_sample_inputs(expert_st, batch_size)).float()

        valid = Variable(FloatTensor(batch_size, 1).fill_(1.), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.), requires_grad=False)
        
        '''generate actions'''
        gen_labels = net.select_action(gen_inputs)
        gather_prob = net.targeting_prob(gen_inputs, gen_labels)
        
        '''updating net'''
        optimizer.zero_grad()
        gen_score = dis(gen_inputs, gen_labels)
        loss = (criterion_dis(gen_score, valid)*gather_prob).mean()
        loss.backward()
        optimizer.step()
        
        '''updating dis'''
        optimizer_dis.zero_grad()
        gen_score = dis(gen_inputs, gen_labels.detach())
        exp_score = dis(inputs, labels)
        loss_dis = (criterion_dis(exp_score, valid).mean() +                     (criterion_dis(gen_score, fake)*gather_prob.detach()).mean())/2.
        loss_dis.backward()
        optimizer_dis.step()
    
        running_loss += loss.item()
        dis_loss += loss_dis.item()
    if epoch%10 == 9:
        print('[{}, {}] generator loss: {} discriminator loss: {}'.format((epoch+1), i+1, running_loss/10, dis_loss/10))
        running_loss = 0.
        dis_loss = 0.

        out_loc = {}
        for i in range(len(test)):
            output = net.select_action(torch.FloatTensor(torch.from_numpy(np.asarray([test_st[i]])).float()))
            x = recover_minmaxscale((test[i][0]), min_x, max_x)
            y = recover_minmaxscale((test[i][1]), min_y, max_y)
            if (x, y) not in out_loc:
                out_loc[(x, y)] = {}
                out_loc[(x, y)][output] = 1
            else:
                if output not in out_loc[(x, y)]:
                    out_loc[(x, y)][output] = 1
                else:
                    out_loc[(x, y)][output] += 1
        target = []
        ground = []
        count = 10
        for key in out_loc:
            o1 = np.zeros(10)
            for a, v in out_loc[key].items():
                o1[a] += v
            o1 /= sum(o1)
            if key in exp_loc:
                o2 = np.zeros(10)
                for b, w in exp_loc[key].items():
                    o2[b] += w
                o2 /= sum(o2)
                target.append(o1)
                ground.append(o2)
                count -= 1
        k, c, kls = cross_entropy(target, ground)
        print('KL:',k, c)
        diff = difference(target, ground)
        if temp_kl > k:
            temp_dis = deepcopy(dis)
            temp_net = deepcopy(net)
            temp_kl = k
            print('record')

