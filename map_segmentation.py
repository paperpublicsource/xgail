#!/usr/bin/env python
# coding: utf-8

# In[1]:


### %pylab inline
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
import matplotlib.cm as cm
import collections
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering


# In[61]:


def MyDBSCAN(D, eps, MinPts, spatial_corr,max_pts = 3, w_mat_ind = False):
    labels = [0]*len(D) 
    C = 0
    
    for P in range(0, len(D)):
        if not (labels[P] == 0):
            continue
        
        if w_mat_ind:#if using LISA, the seed grid should have larger LISA value than eps.
            if abs(spatial_corr[D[P]][D[P]]) < eps:#the seed grid should have larger Ii or Gi score.
                labels[P] = -1
            else:
                # Find all of P's neighboring points.
                NeighborPts = regionQuery(D, P, eps, spatial_corr, w_mat_ind)
                if len(NeighborPts) < MinPts:
                    labels[P] = -1 
                else:
                    C += 1
                    growCluster(D, labels, P, NeighborPts, C, eps, MinPts, spatial_corr, w_mat_ind, max_pts = max_pts)
    # All data has been clustered!
    return labels


# In[62]:


def growCluster(D, labels, P, NeighborPts, C, eps, MinPts, spatial_corr, w_mat_ind, max_pts):
    """
    Grow a new cluster with label `C` from the seed point `P`.
    
    This function searches through the dataset to find all points that belong
    to this new cluster. When this function returns, cluster `C` is complete.
    
    Parameters:
      `D`      - The dataset (a list of vectors)
      `labels` - List storing the cluster labels for all dataset points
      `P`      - Index of the seed point for this new cluster
      `NeighborPts` - All of the neighbors of `P`
      `C`      - The label for this new cluster.  
      `eps`    - Threshold distance
      `MinPts` - Minimum required number of neighbors
      `spatial_corr` - either the spatial correlation or the weight matrix from LISA
      'w_mat_ind' - indicate if spatial_corr is correlation or weight.
    """

    labels[P] = C

    i = 0
    count_C = 0
    max_points_in_C = max_pts
    
    while i < len(NeighborPts):    
        
        # Get the next point from the queue.        
        Pn = NeighborPts[i]
        if labels[Pn] == -1:
            labels[Pn] = C
            count_C += 1
        if labels[Pn] == 0:
            labels[Pn] = C
            count_C += 1
            PnNeighborPts = regionQuery(D, Pn, eps, spatial_corr, w_mat_ind)
            if len(PnNeighborPts) >= MinPts:
                NeighborPts = NeighborPts + PnNeighborPts
          
        i += 1 
        if count_C >= max_points_in_C:
            break


# In[5]:


def regionQuery(D, P, eps, spatial_corr, w_mat_ind):
    """
    Find all points in dataset `D` within distance `eps` of point `P`.
    
    This function calculates the distance between a point P and every other 
    point in the dataset, and then returns only those points which are within a
    threshold distance `eps`.
    """
    neighbors = []
    
    for Pn in range(0, len(D)):
        
        # If the distance is below the threshold, add it to the neighbors list.
        if w_mat_ind: #spatial corr is actually the weight matrix, if weight matrix > eps, expand.
            if abs(spatial_corr[D[P]][D[Pn]]) > eps and _spatial_neighbor(D[P], D[Pn]):
                neighbors.append(Pn)
        else: #spatial_corr is the real spatial correlation, distance should be 1-abs(corr)
            if 1.0-abs(spatial_corr[D[P]][D[Pn]]) < eps and _spatial_neighbor(D[P], D[Pn]):
                neighbors.append(Pn)
    return neighbors


# In[6]:


def _spatial_neighbor(g0_str, g1_str):
    g0 = g0_str.split('|')
    g1 = g1_str.split('|')
    #neighboring 8 grids
    if abs(int(g0[0])-int(g1[0]))<2 and abs(int(g0[1])-int(g1[1]))<2             and abs(int(g0[0])-int(g1[0])) + abs(int(g0[1])-int(g1[1])) != 0:
        return True
    
    else:
        return False


# In[11]:


def Getis_Ord_Gi(s_test):
    '''
    Calculate the Getis-Ord Gi* score for each grid in each channel.
    '''
    Gi_list = []
    for i in range(1,5):
        ch_test = s_test[i]
        # ch_test = ch_test.data.numpy()
        N = len(ch_test)**2
        x_mean = np.mean(ch_test)
        m2 = np.sum(ch_test**2)/N
        S = np.sqrt(m2-x_mean**2)
#         term0 = 0
        bot = 0
        Gi_ch = []
        for i0 in range(15):
            for j0 in range(15):
                real_loc0 = [loc[0] + i0 - 7, loc[1]+ j0 - 7]
                loc0_str = '|'.join(map(str, real_loc0))
                x0 = ch_test[i0, j0]
                bot += (x0-x_mean)**2
                w2_sum = 0
                w_sum = 0
                term0 = 0
                for i1 in range(15):
                    for j1 in range(15):
                        real_loc1 = [loc[0] + i1 - 7, loc[1]+ j1 - 7]
                        loc1_str = '|'.join(map(str, real_loc1))
                        x1 = ch_test[i1, j1]
                        if abs(i0 - i1) < 2 and abs(j0 - j1) < 2 and abs(i0 - i1) + abs(j0 - j1) != 0:
                            w01 = abs(sc_all[i-1][loc0_str][loc1_str])

                        else:
                            w01 = 0.0
                        w_sum += w01
                        w2_sum += w01**2
                        term0 += w01*(x1)
                Gi = (term0 - x_mean*w_sum) / (S*np.sqrt(((N*w2_sum) - w_sum**2) / (N-1)))
                Gi_ch.append(Gi)
        Gi_list.append(np.resize(np.array(Gi_ch),(15,15)))
    return np.array(Gi_list)


# In[12]:


def plot_st(max_st, lable = '', save = False, text = False, v_min = 0.0, v_max = 1.0):
    for ch_i in range(len(max_st)):
        plt.figure(ch_i)
        
        fig, ax = plt.subplots()
#         plt.pcolor(max_st[ch_i],color = 'grey', cmap='Blues' ,vmin=-0.5, vmax = 0.3)#,vmin=0.0, vmax = 1.0
         
        im = ax.matshow(max_st[ch_i], origin='lower', cmap='Blues',vmin=v_min, vmax = v_max)
        plt.colorbar(im)
        if text:
            for (i, j), z in np.ndenumerate(max_st[ch_i]):
                ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', size = 15,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
        if save:
            plt.savefig('./figures/f'+str(ch_i)+lable+'.png',dpi = 200)

def plot_st0(max_st, save = False, lable = 'dist'):
    for ch_i in range(len(max_st)):
        plt.figure(ch_i)
        plt.pcolor(max_st[ch_i],color = 'grey', cmap='Blues')#,vmin=0.0, vmax = 1.0
        plt.colorbar()
        if save:
            plt.savefig('./figures/f'+str(ch_i)+lable+'.png',dpi = 200)


# In[13]:


def _Ii_to_w_mat(loc, Ii_mat):
    '''
    loc: the center location of the 15X15 region, absolute index [grid0, grid1]
    Ii: the LISA score for each grid, relative index
    
    output:
    w_mat_4ch: the weight matrix for each channel, w_mat_4ch[0][grid0][grid1] = weight
    '''
    w_mat_4ch = []
#     loc = [22, 27]
    for ch_i in range(4):
        Ii_ch = Ii_mat[ch_i]
        w_mat = {}
        for i0 in range(15):
            for j0 in range(15):
                loc_0 = '|'.join(map(str,(loc[0]+(i0-7),loc[1]+(j0-7))))
                w_mat[loc_0] = {}
                for i1 in range(15):
                    for j1 in range(15):
                        loc_1 = '|'.join(map(str,(loc[0]+(i1-7),loc[1]+(j1-7))))
                        w_mat[loc_0][loc_1] = Ii_ch[i1][j1]
        w_mat_4ch.append(w_mat)
    return w_mat_4ch


# In[2]:


# cluster via LISA Gi
max_st = mean_st
Gi_mat = Getis_Ord_Gi(max_st)
w_mat_4ch = _Ii_to_w_mat(l, Gi_mat)
eps_4chs = [0.9,0.75,0.45,0.5]
labels_4chs = []
for i in range(4):    
    labels = MyDBSCAN(D, eps = eps_4chs[i], MinPts=2, max_pts = 10, spatial_corr=w_mat_4ch[i], w_mat_ind=True)
    labels_4chs.append(labels)
    a = np.resize(labels,(15,15))
    plot_features_on_map_different_color(loc = l, max_st = a, ch = -1,                         xy_unreach = [], norm_flag= False,                         side_length=15, zoom_in = True, save = False, axis_tick=False, tag = 'clusters_dbscan_LISA_Gi'+str(i))

