# -*- coding: utf-8 -*-
"""
Spyder Editor

These functions compute the multiview pca for the fish data including limb constraints.

List of functions:
    For 3D transformation to include limb constraints for pca
        objective : objective function
        find_linear_transformation : minimises objective
        get_3d_distance : get 3D distance matrix
    For ensembling:
        ensembling_multiview: gets keypoint by keypoint ensembling
    For PCA: 
        multiview_pca_bodyparts: get PCA transformation over all bodyparts
    Other helper:
        pairwise: gives all pairs from a list of elements
        variance_limb: returns variance of limbs across time
        
"""


import os
import numpy as np
import pandas as pd
import sys
from eks.utils import *
from eks.multiview_pca_smoother import *
import matplotlib.pyplot as plt
import tqdm.notebook as tqdm
from scipy.optimize import *
from scipy.interpolate import interp1d
from eks.ensemble_kalman import ensemble, filtering_pass, kalman_dot, smooth_backward
from eks.gradients2 import autograd_loss, autohessian_loss, gradient_distance, hessian_distance, track_distances





def objective(L, q):
    '''
    Objective function to minimise for transformation on labeled data
    Helper : Minimise \sum_ij Var(||L(q_ti-q_tj)||_2)/\sum_k Var(Lq_tk)

    ----------
    L : ndarray (number of latents, number of latents)
        transformation matrix initialisation.
    q : ndarray (number of keypoints, number of timesteps, number of latents)
        latents vector.

    Returns
    -------
    objective

    '''
    n = int(np.sqrt(len(L)))
    L = np.asarray(L).reshape((n,n))
    upper_indices = np.triu_indices(n,1) #offset to diagonal
    # (n-1)n/2 
    L[upper_indices] = np.zeros((n*(n-1)//2)) # constraint upper triangle to zeros
    
    
    # for i in range(q.shape[0]):
    #     d += np.var(L@q[i,:,:].T)
    #     for j in range(q.shape[0]):
    #         if j != i:
    #             s+= np.var(np.linalg.norm(L@(q[i,:,:]-q[j,:,:]).T,axis=0))
    # return s/d
    s = np.zeros(q.shape[1])
    for i in range(q.shape[0]):
        for j in range(q.shape[0]):
            if j>i:
               #  print(L.shape,(q[i,:]-q[j,:]).shape)
                s += np.linalg.norm((q[i,:]-q[j,:])@L,axis=1)
            else:
                pass
    return np.var(s)/np.var(np.linalg.norm(q@L, axis=2))

# L = np.tri(3,3,0)
# objective(L,q)



def find_linear_transformation(q, L_initial):
    '''
    Find the linear transformation in the latent space that keeps variance of limbs almost constant
    
    Parameters
    ----------
      L : ndarray (number of latents, number of latents)
          transformation matrix initialisation.
      q : ndarray (number of keypoints, number of timesteps, number of latents)
          latents vector.
    
      Returns
    -------
        L :
            3D transformation 
    '''
    n = int(np.sqrt(len(L_initial)))
    # Define the optimization problem with the objective function and constraint
    problem = {
        'fun': objective,
        'x0': L_initial,
        'args':q
        }
    
    # Solve the optimization problem
    result = minimize(**problem)
    
    # Get the optimal solution for L
    optimal_L = result.x
    
    return optimal_L.reshape((n,n))
    

def pairwise(t):
    return [(a, b) for idx, a in enumerate(t) for b in t[idx + 1:]]

def variance_limb(L,q,frames_start=0, frames_end=10):
    d = 0
    s = np.zeros((q.shape[0],q.shape[0]))
    for i in range(q.shape[0]):
        d += np.var(L@q[i,:,:].T)
        for j in range(q.shape[0]):
            if j != i:
                s[i,j]= np.var(np.linalg.norm(L@(q[i,frames_start:frames_end,:]-q[j,frames_start:frames_end,:]).T,axis=0))
    return s/d
    

# def variance_limb_plot(num_frames, L, q, pair_list, keypoint_ensemble_list):
#     # pair_list = pairwise(keys)
#     tot_var =  variance_limb(L,q,frames_start =0,frames_end = q.shape[1])
#     var_dict = {}
#     for key_pair in pair_list:
#         i= keypoint_ensemble_list.index(key_pair[0])
#         j=keypoint_ensemble_list.index(key_pair[1])
#         var_list = []
#         for frame in range(num_frames,q.shape[1]):
#             s = variance_limb(L,q,frames_start = frame-num_frames,frames_end=frame)
#             var_list.append(s[i,j]/tot_var[i,j]*100)
#         var_dict[key_pair] = var_list[1:]
#         plt.plot(var_dict[key_pair], label = '{}'.format(key_pair))
#     plt.legend()
#     plt.title('Variance proportion of limb distance over {}'.format(num_frames)+' frames')
#     return var_dict    



def get_3d_distance(q, L, keypoint_ensemble_list, constrained_keypoints_graph):
    '''
    Get 3D distance graph 

    Parameters
    ----------
    q : ndarray (number of keypoints, number of timesteps, number of latents)
        latents vector.
    L : ndarray (number of latents, number of latents)
        transformation matrix initialisation.
    keypoint_ensemble_list : list of strings
        keypoints list.
    constrained_keypoints_graph : list of tuples
        list of tuples of connected limbs.
    num_cameras : int
        number of cameras.

    Returns
    -------
    D : TYPE
        DESCRIPTION.

    '''

    n, T, v = q.shape
    D = np.zeros((T, n,n))
    # print(L.shape, q.shape)
    new_q = (q@L) # (L@np.vstack(q).T).reshape((n,T,v))
    
    # get constrained distances
    for t in range(T):
        for keypair in constrained_keypoints_graph:
            i = keypoint_ensemble_list.index(keypair[0])
            j = keypoint_ensemble_list.index(keypair[1])
            D[t,i,j]= np.linalg.norm(new_q[i,t,:]-new_q[j,t,:])
            D[t,j,i]= np.linalg.norm(new_q[i,t,:]-new_q[j,t,:])
            
    return D




def ensembling_multiview(markers_list_cameras, keypoint_ensemble, smooth_param, quantile_keep_pca, camera_names):
    '''
    Ensembling keypoint by keypoint

    Parameters
    ----------
    markers_list_cameras : list of pd.DataFrames
        each list element is a dataframe of predictions from one ensemble member
    keypoint_ensemble : list of strings
        list of keypoints.
    smooth_param : float
        ranges from .01-2 (smaller values = more smoothing
    quantile_keep_pca : float
        percentage of the points are kept for multi-view PCA (lowest ensemble variance)
    camera_names : list of strings
        camera names
    Returns
    -------
    ndarray:
        scaled predictions
    list:
        index of good frames
    ndarray:
        good scaled frames
    ndarray:
        ensemble variances
    list:
        camera means
    '''

    num_cameras = len(camera_names)
    markers_list_stacked_interp = []
    markers_list_interp = [[] for i in range(num_cameras)]
    for model_id in range(len(markers_list_cameras[0])):
        bl_markers_curr = []
        camera_markers_curr = [[] for i in range(num_cameras)]
        for i in range(markers_list_cameras[0][0].shape[0]):
            curr_markers = []
            for camera in range(num_cameras):
                markers = np.array(markers_list_cameras[camera][model_id].to_numpy()[i, [0, 1]])
                camera_markers_curr[camera].append(markers)
                curr_markers.append(markers)
            bl_markers_curr.append(np.concatenate(curr_markers)) #combine predictions for both cameras
        markers_list_stacked_interp.append(bl_markers_curr)
        for camera in range(num_cameras):
            markers_list_interp[camera].append(camera_markers_curr[camera])
    markers_list_stacked_interp = np.asarray(markers_list_stacked_interp)
    markers_list_interp = np.asarray(markers_list_interp)
    
    keys = [keypoint_ensemble+'_x', keypoint_ensemble+'_y']
    markers_list_cams = [[] for i in range(num_cameras)]
    for k in range(len(markers_list_interp[0])):
        for camera in range(num_cameras):
            markers_cam = pd.DataFrame(markers_list_interp[camera][k], columns = keys)
            markers_list_cams[camera].append(markers_cam)
            
    #compute ensemble median for each camera
    cam_ensemble_preds = []
    cam_ensemble_vars = []
    cam_ensemble_stacks = []
    cam_keypoints_mean_dict = []
    cam_keypoints_var_dict = []
    cam_keypoints_stack_dict = []
    for camera in range(num_cameras):
        cam_ensemble_preds_curr, cam_ensemble_vars_curr, cam_ensemble_stacks_curr, cam_keypoints_mean_dict_curr, cam_keypoints_var_dict_curr, cam_keypoints_stack_dict_curr = ensemble(markers_list_cams[camera], keys)
        cam_ensemble_preds.append(cam_ensemble_preds_curr)
        cam_ensemble_vars.append(cam_ensemble_vars_curr)
        cam_ensemble_stacks.append(cam_ensemble_stacks_curr)
        cam_keypoints_mean_dict.append(cam_keypoints_mean_dict_curr)
        cam_keypoints_var_dict.append(cam_keypoints_var_dict_curr)
        cam_keypoints_stack_dict.append(cam_keypoints_stack_dict_curr)

    #filter by low ensemble variances
    hstacked_vars = np.hstack(cam_ensemble_vars)
    max_vars = np.max(hstacked_vars,1)
    quantile_keep = quantile_keep_pca
    good_frames = np.where(max_vars <= np.percentile(max_vars, quantile_keep))[0]
    
    good_cam_ensemble_preds = []
    good_cam_ensemble_vars = []
    for camera in range(num_cameras):
        good_cam_ensemble_preds.append(cam_ensemble_preds[camera][good_frames])
        good_cam_ensemble_vars.append(cam_ensemble_vars[camera][good_frames])
    
    good_ensemble_preds = np.hstack(good_cam_ensemble_preds)
    good_ensemble_vars = np.hstack(good_cam_ensemble_vars)
    means_camera = []
    for i in range(good_ensemble_preds.shape[1]):
        means_camera.append(good_ensemble_preds[:,i].mean())
    
    ensemble_preds = np.hstack(cam_ensemble_preds)
    ensemble_vars = np.hstack(cam_ensemble_vars)
    ensemble_stacks = np.concatenate(cam_ensemble_stacks,2)
    scaled_ensemble_stacks = remove_camera_means(ensemble_stacks, means_camera)
    good_scaled_ensemble_preds = remove_camera_means(good_ensemble_preds[None,:,:], means_camera)[0]
    scaled_ensemble_preds = remove_camera_means(ensemble_preds[None,:,:], means_camera)[0]
    
    return scaled_ensemble_preds, good_frames, good_scaled_ensemble_preds, ensemble_vars,means_camera


def multiview_pca_bodyparts(scaled_dict,good_preds_dict,good_frames_dict):
    ''' Stacking up ensembled data for PCA
    

    Parameters
    ----------
    scaled_dict : dict
        scaled predictions.
    good_preds_dict : dict
        scaled predictions where only good frames have been kept.
    good_frames_dict : dict
        good frames

    Returns
    -------
    scaled_dict : dict
        scaled predictions
    ensemble_pca : func
        ensemble pca function
    ensemble_ex_var : ndarray
        ensemble variance.
    ensemble_pcs : ndarray
        ensemble pca principle components.
    good_ensemble_pcs : ndarray
        ensemble pca principle components of good observations

    '''
    n, T, v= np.shape(scaled_dict)
    stacked_preds = np.vstack(scaled_dict)
    stacked_good_preds = np.vstack(good_preds_dict)
    good_ensemble_pcs= {key: None for key in range(len(good_frames_dict))}
    # PCA
    
    ensemble_pca, ensemble_ex_var = pca(stacked_good_preds, 3)
    ensemble_pcs = ensemble_pca.transform(stacked_preds)
    # unstack
    ensemble_pcs = ensemble_pcs.reshape(n, T, 3)
    for key in range(len(good_frames_dict)):
        
    
        good_ensemble_pcs[key] = ensemble_pcs[key][good_frames_dict[key]]

    
    return scaled_dict,ensemble_pca,ensemble_ex_var,ensemble_pcs,good_ensemble_pcs
    





# Graph of the forme [('mid','chin'),('fork','chin')]

# if full graph can use pairwise(keypoint_ensemble_list)


# assume y is shaped as y[0,:] [bodypart, camera, coord] 6 [bodypart, camera, coord] 6 ... ex shape (51,18)

def filtering_pass_with_constraint(y, m0, S0, C, R, A, Q, ensemble_vars, D,L, keypoint_ensemble_list,  constrained_keypoints_graph=None, mu=0, loss = 'squared'):
    '''
    

    Parameters
    ----------
    y : ndarray (number of keypoints, number of steps, number of views)
        observations.
    m0 : ndarray (number of latents,)
        initialisation of mean
    S0 : ndarray (number of keypoints, number of latents, number of latents)
        initialisation of variance.
    C : ndarray (number of views, number of latents)
        measurement matrix.
    R : ndarray (number of views, number of views)
        measurement variance.
    A : ndarray (number of latents, number of latents)
        state-transition matrix.
    Q : ndarray (number of latents, number of latents)
        state-covariance.
    ensemble_vars : ndarray 
        ensemble variance.
    D : ndarray (number of keypoints, number of keypoints)
        distance graph.
    L : ndarray (number of latents, number of latents)
        latent space transformation keeping limb approx constant
    keypoint_ensemble_list : list of strings
        keypoint list.
    constrained_keypoints_graph : list of tuples, optional
        connected graph of bodyparts included in regularisation. The default is None which gives all pairwise constraint penalties over bodyparts.
    mu : float64, optional
        regularisation parameter. The default is 0.
    loss : string, optional
        loss type. The default is 'squared'. Other include 'eps' for epsilon insensitive loss, 

    Returns
    -------
    mf : ndarray
        filtering output for the mean.
    Vf : ndarray
        filtering output for the variance.
    S : ndarray
        filtering output for the ensemble variance.

    '''
    
    if constrained_keypoints_graph == None:
        constrained_keypoints_graph = pairwise(keypoint_ensemble_list)
        # all nodes are connected from bodyparts of interest
    # y.shape = (keypoints, time steps, views) 
    T = y.shape[0]  # number of time stpes
    tract_dist = np.zeros((T, len(constrained_keypoints_graph)))
    
    n = len(keypoint_ensemble_list) # number of keypoints
    n_latent=m0.shape[0]//n # number of latents
    v = y.shape[1] # number of views
    #time-varying observation variance
    for i in range(ensemble_vars.shape[1]):
        R[i,i] = ensemble_vars[0][i]
    T = y.shape[0]
    mf = np.zeros(shape=(T, m0.shape[0]))
    Vf = np.zeros(shape=(T, m0.shape[0], m0.shape[0]))
    S = np.zeros(shape=(T, m0.shape[0], m0.shape[0]))
    mf[0] = m0 + kalman_dot(y[0, :] - np.dot(C, m0), S0, C, R)
    Vf[0, :] = S0 - kalman_dot(np.dot(C, S0), S0, C, R)
    S[0] = S0
    
    for i in range(1, T):
        for t in range(ensemble_vars.shape[1]):
            R[t,t] = ensemble_vars[i][t]
        S[i-1] = np.dot(A, np.dot(Vf[i-1, :], A.T)) + Q
        y_minus_CAmf = y[i, :] - np.dot(C, np.dot(A, mf[i-1, :])) 
        
        mf[i, :] = np.dot(A, mf[i-1, :]) + kalman_dot(y_minus_CAmf, S[i-1], C, R)
        Vf[i, :] = S[i-1] - kalman_dot(np.dot(C, S[i-1]), S[i-1], C, R)
        
        if mu!= 0:
            if loss == 'squared':
                grad = gradient_distance(mf[i,:]@L, D, keypoint_ensemble_list, constrained_keypoints_graph)
                # grad = autograd_loss(mf[i,:]@L,  D, keypoint_ensemble_list, constrained_keypoints_graph, mu, loss='squared')
                hess = hessian_distance(mf[i,:]@L, D, keypoint_ensemble_list, constrained_keypoints_graph)
                # hess = autohessian_loss(mf[i,:]@L, D, keypoint_ensemble_list, constrained_keypoints_graph, mu, loss='squared')
            if loss == 'eps':
                grad = autograd_loss(mf[i,:]@L,  D, keypoint_ensemble_list, constrained_keypoints_graph, mu, loss='eps')
                hess = autohessian_loss(mf[i,:]@L,  D, keypoint_ensemble_list, constrained_keypoints_graph, mu, loss='eps')
            
            # add gradient and hessian penalisaiton
            mf[i, :] = np.dot(A, mf[i-1, :]) + kalman_dot(y_minus_CAmf, S[i-1], C, R) + mu*grad
            S[i-1] = np.linalg.inv(np.linalg.inv(S[i-1])+mu*hess[0])
            
        else:
            mf[i, :] = np.dot(A, mf[i-1, :]) + kalman_dot(y_minus_CAmf, S[i-1], C, R) 
            
        Vf[i, :] = S[i-1] - kalman_dot(np.dot(C, S[i-1]), S[i-1], C, R)
        tract_dist[i,:] = track_distances(mf[i,:]@L, keypoint_ensemble_list, constrained_keypoints_graph)
            
    return mf, Vf, S, tract_dist   

    

# def filtering_pass_with_constraint_deprecated(y, m0, S0, C, R, A, Q, ensemble_vars, D,L, keypoint_ensemble_list, constrained_keypoints_graph=None, mu=0.2):
#     if constrained_keypoints_graph == None:
#         constrained_keypoints_graph = pairwise(keypoint_ensemble_list)
#         # all nodes are connected from bodyparts of interest
#     # y.shape = (keypoints, time steps, views) 
#     T = y.shape[1]  # number of time stpes
#     n = len(keypoint_ensemble_list) # number of keypoints
#     v = y.shape[2] # number of views
#     mf = np.zeros(shape=(n,T, m0.shape[0]))
#     Vf = np.zeros(shape=(n,T, m0.shape[0], m0.shape[0]))
#     S = np.zeros(shape=(n,T, m0.shape[0], m0.shape[0]))
#     # for each keypoint
#     for k, part in enumerate(keypoint_ensemble_list):
#         # initial conditions
#         for i in range(v):
#             R[i,i] = ensemble_vars[k][0][i]
#         mf[k,0] =m0 + kalman_dot(y[k,0, :] - np.dot(C, m0), S0[k], C, R)
#         Vf[k,0, :] = S0[k] - kalman_dot(np.dot(C, S0[k]), S0[k], C, R)
#         S[k,0] = S0[k]
#         # filter over time
#     for i in range(1,T):
#         for k, part in enumerate(keypoint_ensemble_list):
#             # ensemble for each camera view
#             for t in range(v):
#                 R[t,t] = ensemble_vars[k][i][t]
#             S[k,i-1] = np.dot(A, np.dot(Vf[k,i-1, :], A.T)) + Q
#             #print(S[i-1], )
#             y_minus_CAmf = y[k,i, :] - np.dot(C, np.dot(A, mf[k,i-1, :]))
           
            
#             if any(part in i for i in constrained_keypoints_graph):
                
#                 # gradient terms
#                 grad = gradient_distance(mf[:,i,:]@L, part, D, keypoint_ensemble_list, constrained_keypoints_graph)
                
#                 hess = hessian_distance(mf[:,i,:]@L, part, D, keypoint_ensemble_list, constrained_keypoints_graph)
#                 # add gradient and hessian penalisaiton
#                 mf[k,i, :] = np.dot(A, mf[k,i-1, :]) + kalman_dot(y_minus_CAmf, S[k,i-1], C, R) + mu*grad
#                 # print(mf[:,i,:].shape)
#                 S[k,i-1] = np.linalg.inv(np.linalg.inv(S[k,i-1])+mu*hess)
                
#             else:
#                 mf[k,i, :] = np.dot(A, mf[k,i-1, :]) + kalman_dot(y_minus_CAmf, S[k,i-1], C, R) 
#             Vf[k,i, :] = S[k,i-1] - kalman_dot(np.dot(C, S[k,i-1]), S[k,i-1], C, R)
            
#     return mf, Vf, S     


#%%%%% TEST 
# tracker_name = 'heatmap_mhcrnn_tracker'
# folder = "/eks_opti"
# operator = "/20210204_Quin/"
# name = "img197707"

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

# image_path = "/Users/clairehe/Documents/GitHub/eks/data/mirror-fish/labeled-data"
# im = plt.imread(image_path+operator+name+".png")
# plt.imshow(im)
# plt.suptitle("labeled "+name)


# session = '20210204_Quin'
# frame = 'img197707.csv'
# smooth_param = 0.01
# quantile_keep_pca = 50
# # Get markers list from networks
# markers_list = []
# for model_dir in model_dirs:
#     csv_file = os.path.join(model_dir, session, frame)
#     df_tmp = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
#     keypoint_names = [l[1] for l in df_tmp.columns[::3]]
#     markers_tmp = convert_lp_dlc(df_tmp, keypoint_names, model_name=tracker_name)
#     markers_list.append(markers_tmp)

# # Ensemble
# scaled_dict = []
# good_frames_dict = []
# good_preds_dict = []
# ensemble_vars_dict = []
# means_camera_dict = []
# for n, keypoint_ensemble in enumerate(keypoint_ensemble_list):
#     markers_list_cameras = [[] for i in range(num_cameras)]
#     for m in markers_list:
#         for camera in range(num_cameras):
#             markers_list_cameras[camera].append(
#                 m[[key for key in m.keys() 
#                     if camera_names[camera] in key 
#                     and 'likelihood' not in key 
#                     and keypoint_ensemble in key]
#                   ]
#             )
#     # ENSEMBLING PER KEYPOINTS
#     scaled_ensemble_preds, good_frames, good_scaled_ensemble_preds,ensemble_vars,means_camera = ensembling_multiview(markers_list_cameras, keypoint_ensemble, smooth_param, quantile_keep_pca, camera_names, plot=True)
#     scaled_dict.append(scaled_ensemble_preds)
#     good_frames_dict.append(good_frames)
#     good_preds_dict.append(good_scaled_ensemble_preds)
#     ensemble_vars_dict.append(ensemble_vars)
#     means_camera_dict.append(means_camera)
# ensemble_vars = np.array(ensemble_vars_dict)


# stacked_preds,ensemble_pca,ensemble_ex_var,ensemble_pcs,good_ensemble_pcs =  multiview_pca_bodyparts(scaled_dict,good_preds_dict,good_frames_dict)

# y_obs = np.asarray(stacked_preds)

# #compute center of mass
# #latent variables (observed)
# good_z_t_obs = good_ensemble_pcs #latent variables - true 3D pca

# n, T, v = y_obs.shape

# ##### Set values for kalman filter #####
# m0 = np.asarray([0.0, 0.0, 0.0]) # initial state: mean
# S0 = np.zeros((nkeys,m0.shape[0], m0.shape[0] ))
# d_t = {key: None for key in range(nkeys)}
# # need different variance for each bodyparts 
# for k in range(n):
#     S0[k,:,:] =  np.asarray([[np.var(good_z_t_obs[k][:,0]), 0.0, 0.0], [0.0, np.var(good_z_t_obs[k][:,1]), 0.0], [0.0, 0.0, np.var(good_z_t_obs[k][:,2])]]) # diagonal: var
#     d_t[k] = good_z_t_obs[k][1:] - good_z_t_obs[k][:-1]

#     Q = smooth_param*np.cov(d_t[k].T)

# A = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) #state-transition matrix,
# # Q = np.asarray([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]) #state covariance matrix?????


# C = ensemble_pca.components_.T # Measurement function is inverse transform of PCA
# R = np.eye(ensemble_pca.components_.shape[1]) # placeholder diagonal matrix for ensemble variance

# print(f"filtering ...")
# mfc, Vfc, Sc = filtering_pass_with_constraint(y_obs, m0, S0, C, R, A, Q,ensemble_vars, D, L,keypoint_ensemble_list, constrained_keypoints_graph=[('fork','mid'),('chin_base','fork')], mu=0)


# # Do the smoothing step
# print("done filtering")
# y_m_filt = {key: None for key in range(n)}
# y_v_filt = {key: None for key in range(n)}
# y_m_smooth = {key: None for key in range(n)}
# y_v_smooth = {key: None for key in range(n)}
# ms = {key: None for key in range(n)}
# Vs = {key: None for key in range(n)}
# for k in range(n):
#     y_m_filt[k] = np.dot(C, mfc[k].T).T
#     y_v_filt[k] = np.swapaxes(np.dot(C, np.dot(Vfc[k], C.T)), 0, 1)
#     print(f"smoothing {keypoint_ensemble_list[k]}...")
#     ms[k], Vs[k], _ = smooth_backward(y_obs[k], mfc[k], Vfc[k], Sc[k], A, Q, C)

#     print("done smoothing")

#     # Smoothed posterior over yb
#     y_m_smooth[k] = np.dot(C, ms[k].T).T
#     y_v_smooth[k] = np.swapaxes(np.dot(C, np.dot(Vs[k], C.T)), 0, 1)












