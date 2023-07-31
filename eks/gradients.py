#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 05:49:53 2023

@author: claire he

These functions compute either gradients and hessian steps of limb distance constraints either autograd for a given loss.

List of functions:
    squared loss: squared loss penalty for constraint
    eps insensitive loss: epsilon insensitive loss penalty for constraint
    gradient_distance : for squared loss, derived from the equations
    hessian_distance : for squared loss, derived from equations
    autograd_loss : auto gradient, can support losses with specifications
    autohessian_loss : auto hessian, as above. 

"""

import os
import autograd.numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import tqdm.notebook as tqdm
from scipy.optimize import *
from scipy.interpolate import interp1d
from autograd import * 



#%%% SQUARED LOSS



def squared_loss(q, D, keypoint_ensemble_list, constrained_keypoints_graph, mu): 
    '''
        Squared loss penalty on the distance of limb constraint
        mu* sum (|qtj - q_tk|_2 - D_jk)^2 
        
    q : (number of keypoints, number of time steps, number of latents)
        vector of latents
    D : (number of keypoints, number of keypoints)
        matrice of distances taken over labeled frames
    keypoint_ensemble_list : (number of keypoints, )
        keypoint list of all bodyparts involved
    constrained_keypoint_graph : 
        list of tuples with keypoints connected with a limb distance constraint
    mu : tuning parameter of squared loss penalty
    -----
    returns loss 
    
    '''
    
    # sum (|qtj - q_tk|_2 - D_jk)^2
    loss = [0 for p in range(len(keypoint_ensemble_list))]
    for p, part in enumerate(keypoint_ensemble_list):
        nei_idx = []
        neighbors = [item[0] for item in constrained_keypoints_graph if item[1] == part]+[item[1] for item in constrained_keypoints_graph if item[0] == part]
        # print(neighbors)
        for elem in neighbors:
            nei_idx.append(keypoint_ensemble_list.index(elem))  # get neighbor index
        
        if neighbors == []:
            loss[p] = 0
        
        else:
            for idx in nei_idx:
                # print('index',np.sum((np.linalg.norm(q[p,:,:]-q[idx,:,:],axis=1)-D[p,idx])**2),loss[p])
                loss[p] += np.sum((np.linalg.norm(q[p,:]-q[idx,:],axis=0)-D[p,idx])**2)
                
    return mu*np.sum(loss)

def eps_insensitive_loss(q,D, keypoint_ensemble_list, constrained_keypoints_graph, mu):
    '''
        Epsilon insensitive loss penalty on the distance of limb constraint
        mu* sum max(0, ||qtj-qtk|-Djk| - eps) 
                    where eps has be set to be Var(D_jk) for now 
        
        
    q : (number of keypoints, number of time steps, number of latents)
        vector of latents
    D : (number of keypoints, number of keypoints)
        matrice of distances taken over labeled frames
    keypoint_ensemble_list : (number of keypoints, )
        keypoint list of all bodyparts involved
    constrained_keypoint_graph : 
        list of tuples with keypoints connected with a limb distance constraint
    mu : tuning parameter of loss penalty
    
    -----
    returns loss
    '''
    
    
    eps = np.sqrt(np.var(D))
    loss = [0 for p in range(len(keypoint_ensemble_list))]
    for p, part in enumerate(keypoint_ensemble_list):
        nei_idx = []
        neighbors = [item[0] for item in constrained_keypoints_graph if item[1] == part]+[item[1] for item in constrained_keypoints_graph if item[0] == part]
        # print(neighbors)
        for elem in neighbors:
            nei_idx.append(keypoint_ensemble_list.index(elem))  # get neighbor index
        
        if neighbors == []:
            loss[p] = 0
        
        else:
            for idx in nei_idx:
                # print('index',np.sum((np.linalg.norm(q[p,:,:]-q[idx,:,:],axis=1)-D[p,idx])**2),loss[p])
                loss[p] += np.sum(max(0.0, np.abs(np.linalg.norm(q[p,:]-q[idx,:],axis=0)-D[p,idx]) - eps))
                
    return mu*np.sum(loss)



def gradient_distance(q, part, D, keypoint_ensemble_list, constrained_keypoints_graph):
    '''
        Squared loss gradient 
            sum_nodes connected to part (q[part,:]-q[connected_part,:])/np.linalg.norm(----) 
    
    ----------
    q : (number of keypoints, number of time steps, number of latents)
        vector of latent
    part : string 
        bodypart being calculated
    D : (number of keypoints, number of keypoints)
        distance graph between keypoints
    keypoint_ensemble_list : (number of keypoints,)
        all keypoints list
    constrained_keypoints_graph : list of tuples
        list of tuples giving the graph of connected components in the bodyparts.

    Returns
    -------
    gradient 

    '''
    
    p = keypoint_ensemble_list.index(part)
    n,v = q.shape
    neighbors = [item[0] for item in constrained_keypoints_graph if item[1] == part]+[item[1] for item in constrained_keypoints_graph if item[0] == part]
    nei_idx = []
    grad = np.zeros(v)
    for elem in neighbors:
        nei_idx.append(keypoint_ensemble_list.index(elem))  # get neighbor index
    for idx in nei_idx:
        if (np.linalg.norm(q[p, :] - q[idx, :])>0):
            grad += (q[p, :] - q[idx, :])*(1 - D[p,idx]/(np.linalg.norm(q[p, :] - q[idx, :])))
    return 2*grad
    
    
def hessian_distance(q, part,D, keypoint_ensemble_list, constrained_keypoints_graph):
    '''
        Hessian of squared loss
        
    ----------
    q : (number of keypoints, number of time steps, number of latents)
        vector of latent
    part : string 
        bodypart being calculated
    D : (number of keypoints, number of keypoints)
        distance graph between keypoints
    keypoint_ensemble_list : (number of keypoints,)
        all keypoints list
    constrained_keypoints_graph : list of tuples
        list of tuples giving the graph of connected components in the bodyparts.

    Returns
    -------
    hessian
    '''
    p = keypoint_ensemble_list.index(part)
    neighbors = [item[0] for item in constrained_keypoints_graph if item[1] == part]+[item[1] for item in constrained_keypoints_graph if item[0] == part]
    nei_idx = []
    n = len(keypoint_ensemble_list)
    hess = np.zeros((n,n))
    for elem in neighbors:
        nei_idx.append(keypoint_ensemble_list.index(elem))  # get neighbor index
    
    for idx in nei_idx:
        if (np.linalg.norm(q[p, :] - q[idx, :])>0):
            
            hess += -(np.eye(n)-D[p,idx]* (np.eye(n)/(np.linalg.norm(q[p, :] - q[idx, :]))@ \
                        (np.eye(n) - 1/(np.linalg.norm(q[p, :] - q[idx, :])**2)*(q[p,:]-q[idx,:])@(q[p,:]-q[idx,:]).T)))
    return 2*hess



def autograd_loss(x,  D_ij, keypoint_ensemble_list, constrained_keypoints_graph, mu, loss='squared'):
    '''

    Function that returns the autograd from a given loss 
    from list : 'squared', 'eps'
    ----------
    x : (number of keypoints, number of time steps, number of latents)
        vector of latents.
    D_ij : (number of keypoints, number of keypoints)
        distance graph of bodyparts.
    keypoint_ensemble_list : (number of keypoints,)
        list of keypoints
    constrained_keypoints_graph : list of tuples
        list of tuples giving the graph of connected components in the bodyparts.
    mu : float64
        regularisation parameter
    loss : string, optional
        loss type. The default is 'squared'.

    Returns
    -------
    gradient of chosen loss.

    '''
    if loss == 'squared':
        gr = grad(lambda q: squared_loss(q, D_ij, keypoint_ensemble_list, constrained_keypoints_graph, mu))
    elif loss == 'eps':
        gr =  grad(lambda q: eps_insensitive_loss(q, D_ij, keypoint_ensemble_list, constrained_keypoints_graph, mu))
    return(gr(x)+0.001)

def autohessian_loss(x, D_ij, keypoint_ensemble_list, constrained_keypoints_graph, mu, loss = 'squared'):
    '''

 
    Function that returns the hessian from a given loss 
    from list : 'squared', 'eps'
    ----------
    x : (number of keypoints, number of time steps, number of latents)
        vector of latents.
    D_ij : (number of keypoints, number of keypoints)
        distance graph of bodyparts.
    keypoint_ensemble_list : (number of keypoints,)
        list of keypoints
    constrained_keypoints_graph : list of tuples
        list of tuples giving the graph of connected components in the bodyparts.
    mu : float64
        regularisation parameter
    loss : string, optional
        loss type. The default is 'squared'.

    Returns
    -------
    hessian of chosen loss.
    '''
    if loss == 'squared':
        h = hessian(lambda q: squared_loss(q, D_ij, keypoint_ensemble_list, constrained_keypoints_graph, mu))
    elif loss == 'eps':
        h = hessian(lambda q: eps_insensitive_loss(q, D_ij, keypoint_ensemble_list, constrained_keypoints_graph, mu))
    return h(x)
    






# camera_names = ['main', 'top', 'right']
# keypoint_ensemble_list = ['mid','fork','chin_base']
# #keypoint_ensemble_list = [ 'head', 'chin_base', 'chin1_4', 'chin_half','chin3_4', 'chin_tip', 'mid', 'fork',
# # 'stripeA', 'stripeP', 'tail_neck', 'dorsal', 'anal', 'caudal_d', 'caudal_v']
# tracker_name = 'heatmap_mhcrnn_tracker'
# num_cameras = len(camera_names)
# labeled_data = pd.read_csv("/Users/clairehe/Documents/GitHub/eks/data/mirror-fish/CollectedData_new.csv", header = [1,2])



# mu = [0,0.005,0.001]
# c = [('fork','chin_base'),('fork', 'mid'), ('chin_base','mid')]


# session = '20210204_Quin'
# folder = "/eks_opti"
# operator = "/20210204_Quin/"
# #name = "img048416" 
# name =  "img197707" 
# frame = name+'.csv'

# baseline = pd.read_csv("/Users/clairehe/Documents/GitHub/eks/data/misc/mirror-fish_ensemble-predictions/eks"+operator+name+".csv", header=[ 1, 2],index_col=0)
# #new = pd.read_csv("/Users/clairehe/Documents/GitHub/eks/data/misc/one-video-mirror-fish-predictions"+folder+operator+name, header=[ 1, 2], index_col=0)
# baseline0 = pd.read_csv("/Users/clairehe/Documents/GitHub/eks/data/misc/mirror-fish_ensemble-predictions/eks"+operator+name+".csv", header=[0, 1, 2],index_col=0)


# # NOTE! replace this path with an absolute path where you want to save EKS outputs
# eks_save_dir = '/Users/clairehe/Documents/GitHub/eks/data/misc/one-video-mirror-fish-predictions/eks_opti/'

# # path for prediction csvs
# file_path = '/Users/clairehe/Documents/GitHub/eks/data/misc/mirror-fish_ensemble-predictions'

# # NOTE! replace these paths with the absolute paths to prediction csvs on your local computer
# model_dirs = [
#     file_path+"/network_0",
#     file_path+"/network_1",
#     file_path+"/network_2",
#     file_path+"/network_3",
#     file_path+"/network_4",
# ]


# #    'head', 'chin_base', 'chin1_4', 'chin_half','chin3_4', 'chin_tip', 'mid', 'fork',
# #   'stripeA', 'stripeP', 'tail_neck', 'dorsal', 'anal', 'caudal_d', 'caudal_v',


# smooth_param = 0.01
# quantile_keep_pca = 50

# camkeys = ["_main","_top","_right"]
# # flatten columns
# labeled_data.columns = ['_'.join(tup).rstrip('_') for tup in labeled_data.columns.values]

# img_id = labeled_data.loc[['labeled-data'+operator+name+'.png' in s for s in labeled_data.bodyparts_coords]].index[0]



# #%% set distance constraint projection for a few q's in keypoint_ensemble_list



# mask = []
# for keys in keypoint_ensemble_list:
#     for cam in camkeys:
#         for coord in ["_x", "_y"]:
#             mask.append(keys+cam+coord)
            


# markers_list = labeled_data.reset_index()[mask]
# # Ensemble
# scaled_dict = []
# good_frames_dict = []
# good_preds_dict = []
# ensemble_vars_dict = []
# markers_list_cameras  = []
            
# num_mar = len(markers_list)
# n = len(keypoint_ensemble_list)
# num_cameras = len(camkeys)
# y_obs = np.empty((n, num_mar, 2*num_cameras))
# q = np.empty((n, num_mar, 3))


# for j, keypoint_ensemble in enumerate(keypoint_ensemble_list):
#     markers_list_cameras = [[] for i in range(num_cameras)]
#     for i,cam in enumerate(camkeys):
#         tmp = []

#         for m in markers_list.keys():
#             if cam in m and keypoint_ensemble in m:
#                 tmp.append(markers_list[m])
    
#         markers_list_cameras[i].append(pd.concat(tmp, axis=1))

    
#     y = np.asarray(markers_list_cameras).reshape((num_mar, 2*num_cameras))
    
#     # fill nans by median value 
#     col_mean = np.nanmedian(y, axis=0)
#     inds = np.where(np.isnan(y))
#     y[inds] = np.take(col_mean, inds[1])
#     means_camera = np.mean(y,axis=0)
    
#     # scale 
#     y -= means_camera
#     # scaled_y = scale(y)
#     # get PCA 
#     labeled_pca, labeled_var = pca(y, 3)
    
#     q[j,:,:] = labeled_pca.transform(y)
#     y_obs[j,:,:] = y

#   # Define the size of L
# L_initial = np.tril(np.eye(3)).flatten()
# L = find_linear_transformation(q, L_initial)

# D_ij = get_3d_distance_loss(q, L, keypoint_ensemble_list, c, num_cameras)[img_id]


# c = [('fork','mid'),('fork','chin_base')]


