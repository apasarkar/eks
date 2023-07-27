#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 08:27:45 2023

@author: clairehe
"""

from tqdm import tqdm
import cv2
import os
from eks.multiview_pca_fish import *
from eks.video_utils import *

camera_names = ['main', 'top', 'right']
keypoint_ensemble_list = ['mid','fork','chin_base']
#keypoint_ensemble_list = [ 'head', 'chin_base', 'chin1_4', 'chin_half','chin3_4', 'chin_tip', 'mid', 'fork',
# 'stripeA', 'stripeP', 'tail_neck', 'dorsal', 'anal', 'caudal_d', 'caudal_v']
tracker_name = 'heatmap_mhcrnn_tracker'
num_cameras = len(camera_names)
labeled_data = pd.read_csv("/Users/clairehe/Documents/GitHub/eks/data/mirror-fish/CollectedData_new.csv", header = [1,2])



mu = [0,0.005,0.001]
c = [('chin_base','mid')]


#baseline = pd.read_csv("/Users/clairehe/Documents/GitHub/eks/data/misc/mirror-fish_ensemble-predictions/eks"+operator+name+".csv", header=[ 1, 2],index_col=0)
#new = pd.read_csv("/Users/clairehe/Documents/GitHub/eks/data/misc/one-video-mirror-fish-predictions"+folder+operator+name, header=[ 1, 2], index_col=0)
#baseline0 = pd.read_csv("/Users/clairehe/Documents/GitHub/eks/data/misc/mirror-fish_ensemble-predictions/eks"+operator+name+".csv", header=[0, 1, 2],index_col=0)


# NOTE! replace this path with an absolute path where you want to save EKS outputs
eks_save_dir = '/Users/clairehe/Documents/GitHub/eks/data/misc/one-video-mirror-fish-predictions/eks_opti/'

# path for prediction csvs
file_path = '/Users/clairehe/Documents/GitHub/eks/data/misc/mirror-fish_ensemble-predictions'

# NOTE! replace these paths with the absolute paths to prediction csvs on your local computer
model_dirs = [
    file_path+"/network_0",
    file_path+"/network_1",
    file_path+"/network_2",
    file_path+"/network_3",
    file_path+"/network_4",
]


#    'head', 'chin_base', 'chin1_4', 'chin_half','chin3_4', 'chin_tip', 'mid', 'fork',
#   'stripeA', 'stripeP', 'tail_neck', 'dorsal', 'anal', 'caudal_d', 'caudal_v',


smooth_param = 0.01
quantile_keep_pca = 50

camkeys = ["_main","_top","_right"]
# flatten columns
labeled_data.columns = ['_'.join(tup).rstrip('_') for tup in labeled_data.columns.values]



mask = []
for keys in keypoint_ensemble_list:
    for cam in camkeys:
        for coord in ["_x", "_y"]:
            mask.append(keys+cam+coord)
            


markers_list = labeled_data.reset_index()[mask]
# Ensemble
scaled_dict = []
good_frames_dict = []
good_preds_dict = []
ensemble_vars_dict = []
markers_list_cameras  = []
            
num_mar = len(markers_list)
n = len(keypoint_ensemble_list)
num_cameras = len(camkeys)
y_obs = np.empty((n, num_mar, 2*num_cameras))
q = np.empty((n, num_mar, 3))


for j, keypoint_ensemble in enumerate(keypoint_ensemble_list):
    markers_list_cameras = [[] for i in range(num_cameras)]
    for i,cam in enumerate(camkeys):
        tmp = []

        for m in markers_list.keys():
            if cam in m and keypoint_ensemble in m:
                tmp.append(markers_list[m])
    
        markers_list_cameras[i].append(pd.concat(tmp, axis=1))

    
    y = np.asarray(markers_list_cameras).reshape((num_mar, 2*num_cameras))
    
    # fill nans by median value 
    col_mean = np.nanmedian(y, axis=0)
    inds = np.where(np.isnan(y))
    y[inds] = np.take(col_mean, inds[1])
    means_camera = np.mean(y,axis=0)
    
    # scale 
    y -= means_camera
    # scaled_y = scale(y)
    # get PCA 
    labeled_pca, labeled_var = pca(y, 3)
    
    q[j,:,:] = labeled_pca.transform(y)
    y_obs[j,:,:] = y

  # Define the size of L
L_initial = np.tril(np.eye(3)).flatten()
L = find_linear_transformation(q, L_initial)


sessions = os.listdir(model_dirs[0])

folder = "/eks_opti"
#operator = "/20210204_Quin/"
#name = "img048416" 
#name =  "img197707" 
#frame = name+'.csv'

for session in sessions:
    operator= '/'+session+'/'
    frames = os.listdir(os.path.join(model_dirs[0], session))
    for frame in frames:
        name = frame[:-4]
        img_id = labeled_data.loc[['labeled-data'+operator+name+'.png' in s for s in labeled_data.bodyparts_coords]].index[0]
        D_ij = get_3d_distance_loss(q, L, keypoint_ensemble_list, c, num_cameras)[img_id]
    
    
        # Get markers list from networks
        markers_list = []
        for model_dir in model_dirs:
            csv_file = os.path.join(model_dir, session, frame)
            df_tmp = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
            keypoint_names = [l[1] for l in df_tmp.columns[::3]]
            markers_tmp = convert_lp_dlc(df_tmp, keypoint_names, model_name='heatmap_mhcrnn_tracker')
            markers_list.append(markers_tmp)
        
        # Ensemble
        scaled_dict = []
        good_frames_dict = []
        good_preds_dict = []
        ensemble_vars_dict = []
        means_camera_dict = []
        for n, keypoint_ensemble in enumerate(keypoint_ensemble_list):
            markers_list_cameras = [[] for i in range(num_cameras)]
            for m in markers_list:
                for camera in range(num_cameras):
                    markers_list_cameras[camera].append(
                        m[[key for key in m.keys() 
                           if camera_names[camera] in key 
                           and 'likelihood' not in key 
                           and keypoint_ensemble in key]
                         ]
                    )
            # ENSEMBLING PER KEYPOINTS
            scaled_ensemble_preds, good_frames, good_scaled_ensemble_preds,ensemble_vars,means_camera = ensembling_multiview(markers_list_cameras, keypoint_ensemble, smooth_param, quantile_keep_pca, camera_names, plot=True)
            scaled_dict.append(scaled_ensemble_preds)
            good_frames_dict.append(good_frames)
            good_preds_dict.append(good_scaled_ensemble_preds)
            ensemble_vars_dict.append(ensemble_vars)
            means_camera_dict.append(means_camera)
        ensemble_vars = np.array(ensemble_vars_dict)
        
        
        stacked_preds,ensemble_pca,ensemble_ex_var,ensemble_pcs,good_ensemble_pcs =  multiview_pca_bodyparts(scaled_dict,good_preds_dict,good_frames_dict)
        
        y_obs = np.asarray(stacked_preds)
        
        
        
        #compute center of mass
        #latent variables (observed)
        good_z_t_obs = good_ensemble_pcs #latent variables - true 3D pca
        
        
        ## set L
        #L_initial = np.tril(np.eye(3)).flatten()
        #L = find_linear_transformation(np.asarray([list(good_z_t_obs.items())[i][1] for i in range(len(list(good_z_t_obs.items())[0]))]), L_initial)
        
        
        n, T, v = y_obs.shape
        #print(y_obs.shape)
        ##### Set values for kalman filter #####
        m0 = np.zeros(3) # initial state: mean
        S0 = np.zeros((n,m0.shape[0], m0.shape[0] ))
        d_t = {key: None for key in range(n)}
        # need different variance for each bodyparts 
        for k in range(n):
            S0[k,:,:] =  np.asarray([[np.var(good_z_t_obs[k][:,0]), 0.0, 0.0], [0.0, np.var(good_z_t_obs[k][:,1]), 0.0], [0.0, 0.0, np.var(good_z_t_obs[k][:,2])]]) # diagonal: var
            d_t[k] = good_z_t_obs[k][1:] - good_z_t_obs[k][:-1]
        
            Q = smooth_param*np.cov(d_t[k].T)
        
        A = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) #state-transition matrix,
        # Q = np.asarray([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]) #state covariance matrix?????
        
        
        C = ensemble_pca.components_.T # Measurement function is inverse transform of PCA
        R = np.eye(ensemble_pca.components_.shape[1]) # placeholder diagonal matrix for ensemble variance
        
        all_mu = {0:None, 1: None}
        #print(f"filtering ...")
        for i in range(len(mu)):
            mfc, Vfc, Sc  = filtering_pass_with_constraint(y_obs, m0, S0, C, R, A, Q,ensemble_vars, D_ij, L, keypoint_ensemble_list, constrained_keypoints_graph=c, mu=mu[i])
        
            ## Do the smoothing step
            #print("done filtering")
            y_m_filt = {key: None for key in range(n)}
            y_v_filt = {key: None for key in range(n)}
            y_m_smooth = {key: None for key in range(n)}
            y_v_smooth = {key: None for key in range(n)}
            ms = {key: None for key in range(n)}
            Vs = {key: None for key in range(n)}
            for k in range(n):
                y_m_filt[k] = np.dot(C, mfc[k].T).T
                y_v_filt[k] = np.swapaxes(np.dot(C, np.dot(Vfc[k], C.T)), 0, 1)
                #print(f"smoothing {keypoint_ensemble_list[k]}...")
                ms[k], Vs[k], _ = smooth_backward(y_obs[k], mfc[k], Vfc[k], Sc[k], A, Q, C)
                #print("done smoothing")
        
                # Smoothed posterior over yb
                y_m_smooth[k] = np.dot(C, ms[k].T).T
                y_v_smooth[k] = np.swapaxes(np.dot(C, np.dot(Vs[k], C.T)), 0, 1)
        
            ncams = len(camkeys)
            camera_dfs = {}
            camera_indices = []
            for camera in range(ncams):
                camera_indices.append([camera*2, camera*2+1])
        
            for camera, camera_name in enumerate(camera_names):
                key_df = []
                for k, keypoint_ensemble in enumerate(keypoint_ensemble_list):
                    pdindex = make_dlc_pandas_index([keypoint_ensemble])
                    var = np.empty(y_m_smooth[k].T[camera_indices[camera][0]].shape)
                    var[:] = np.nan
                    pred_arr = pd.DataFrame(np.vstack([
                        y_m_smooth[k].T[camera_indices[camera][0]] + means_camera_dict[k][camera_indices[camera][0]],
                        y_m_smooth[k].T[camera_indices[camera][1]] + means_camera_dict[k][camera_indices[camera][1]],
                        var,
                    ]).T, columns = pdindex)
                    key_df.append(pred_arr)
                #print(means_camera[camera_indices[camera][0]],means_camera[camera_indices[camera][1]])
                camera_dfs[camera_name + '_df'] = pd.concat(key_df,axis=1)
        
            all_mu[i] = camera_dfs
            
            
    
        
        base_path = '/Users/clairehe/Documents/GitHub/eks/data/mirror-fish/videos-for-each-labeled-frame'
        
        
        save_file = os.path.join(base_path, f'%s_eks.mp4' % name)
        framerate=1
        
        cap = cv2.VideoCapture(base_path+operator+name+'.mp4')
        
        marker_shapes = ['D','o', '^', 's','v'] # 4 models 
        colors = ['red', 'green', 'blue'] # 3 views
        
        single_alpha = .5
        single_alpha_list = [single_alpha]*len(all_mu)
        alphas = single_alpha_list + [1.0]
        model_labels = ['ensemble', f'kalman, E:{mu}']
        frame_idxs = [i for i in range(len(all_mu[0]['main_df']['ensemble-kalman_tracker']['mid']['x']))]
        tmp_dir = os.path.join(os.path.dirname(save_file), 'tmpZzZ')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        # plot 
        fig, ax = plt.subplots(1,1,figsize=(10, 10))
        txt_fr_kwargs = {
            'fontsize': 14, 'color': [1, 1, 1], 'horizontalalignment': 'left',
            'verticalalignment': 'top', 'fontname': 'monospace',
            'bbox': dict(facecolor='k', alpha=0.25, edgecolor='none'),
            'transform': ax.transAxes
        }
        for idx in frame_idxs:
            # important!! otherwise each frame will plot on top of the last
            ax.clear()
        
            frame = get_frames_from_idxs(cap, [idx])
            # plot original frame
            ax.imshow(frame[0, 0], vmin=0, vmax=255, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
        
            patches = []
            plot_video_markers(all_mu, keypoint_ensemble_list, camera_names, ax, idx, marker_shapes, colors, alphas, mu)
            plt.legend(mu)
            plt.title(c)
            im = ax.text(0.02, 0.98, 'frame %i' % idx, **txt_fr_kwargs)
            plt.savefig(os.path.join(tmp_dir, 'frame_%06i.jpeg' % idx))
        save_video(save_file, tmp_dir, framerate, frame_pattern='frame_%06i.jpeg')
