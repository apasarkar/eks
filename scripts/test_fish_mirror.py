#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 02:33:33 2023

@author: clairehe

Script to run over fish datasets 

"""
import cv2

from eks.multiview_pca_fish import *
from eks.video_utils import *
from collections import defaultdict


camera_names = ['main', 'top', 'right']
keypoint_ensemble_list = ['mid','fork','chin_base']
#keypoint_ensemble_list = [ 'head', 'chin_base', 'chin1_4', 'chin_half','chin3_4', 'chin_tip', 'mid', 'fork',
# 'stripeA', 'stripeP', 'tail_neck', 'dorsal', 'anal', 'caudal_d', 'caudal_v']
tracker_name = 'heatmap_mhcrnn_tracker'
num_cameras = len(camera_names)
labeled_data = pd.read_csv("/Users/clairehe/Documents/GitHub/eks/data/mirror-fish/CollectedData_new.csv", header = [1,2])



mu = [0,0.1, 0.5, 1]
c = [('fork','chin_base'),('fork', 'mid'), ('chin_base','mid')]


session = '20210204_Quin'
folder = "/eks_opti"
operator = "/20210204_Quin/"
name = "img048416" 
#name =  "img197707" 
frame = name+'.csv'

baseline = pd.read_csv("/Users/clairehe/Documents/GitHub/eks/data/misc/mirror-fish_ensemble-predictions/eks"+operator+name+".csv", header=[ 1, 2],index_col=0)
#new = pd.read_csv("/Users/clairehe/Documents/GitHub/eks/data/misc/one-video-mirror-fish-predictions"+folder+operator+name, header=[ 1, 2], index_col=0)
baseline0 = pd.read_csv("/Users/clairehe/Documents/GitHub/eks/data/misc/mirror-fish_ensemble-predictions/eks"+operator+name+".csv", header=[0, 1, 2],index_col=0)


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

img_id = labeled_data.loc[['labeled-data'+operator+name+'.png' in s for s in labeled_data.bodyparts_coords]].index[0]



#%% set distance constraint projection for a few q's in keypoint_ensemble_list



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

D_ij = get_3d_distance(q, L, keypoint_ensemble_list, c, num_cameras)[img_id]

def variance_plot(L,q):
    s = 0
    d = 0
    n = int(np.sqrt(len(L)))
    #L = np.asarray(L).reshape((n,n))
    upper_indices = np.triu_indices(n,1) #offset to diagonal
    # (n-1)n/2 
    L[upper_indices] = np.zeros((n*(n-1)//2)) # constraint upper triangle to zeros
    for i in range(q.shape[0]):
        d += np.var(L@q[i,:,:].T)
        for j in range(q.shape[0]):
            if j != i:
                s+= np.linalg.norm(L@(q[i,:,:]-q[j,:,:]).T,axis=0)
                # print(keypoint_ensemble_list[i], keypoint_ensemble_list[j], i)
    return s/d
color = ['blue','red']
s = [variance_plot(L,q), variance_plot(np.eye(3), q)]
leg = ['variance obtained','variance in pca']
fig, ax = plt.subplots(1,2,figsize=(20,6))
#ax[0].set(ylim=(0.001305, 0.001545))
#ax[1].set(ylim=(0.00456, 0.00468))
for i in range(2):
    ax[i].errorbar([j for j in range(len(s[i]))], y = [np.mean(s[i]) for j in range(len(s[i]))],
           yerr=np.sqrt((s[i]-np.mean(s[i]))**2), fmt='o', color=color[i], label=leg[i])
    ax[i].set_xlabel('x-axis')
    ax[i].set_ylabel('y-axis')

    ax[i].set_title('variance plot')
    ax[i].legend()

#%%%%% REPROCESSING




def filtering_pass_with_constraint2(y, m0, S0, C, R, A, Q, ensemble_vars, D, keypoint_ensemble_list, constrained_keypoints_graph=None, mu=0.2):
    if constrained_keypoints_graph == None:
        constrained_keypoints_graph = pairwise(keypoint_ensemble_list)
        # all nodes are connected from bodyparts of interest
    # y.shape = (keypoints, time steps, views) 
    T = y.shape[1]  # number of time stpes
    n = len(keypoint_ensemble_list) # number of keypoints
    v = y.shape[2] # number of views
    mf = np.zeros(shape=(n,T, m0.shape[0]))
   
    Vf = np.zeros(shape=(n,T, m0.shape[0], m0.shape[0]))
    S = np.zeros(shape=(n,T, m0.shape[0], m0.shape[0]))
    # for each keypoint
    for k, part in enumerate(keypoint_ensemble_list):
        # initial conditions
        for i in range(v):
            R[i,i] = ensemble_vars[k][0][i]
        mf[k,0] =m0 + kalman_dot(y[k,0, :] - np.dot(C, m0), S0[k], C, R)
        Vf[k,0, :] = S0[k] - kalman_dot(np.dot(C, S0[k]), S0[k], C, R)
        S[k,0] = S0[k]
        # filter over time
    for i in range(1,T):
        for k, part in enumerate(keypoint_ensemble_list):
            # ensemble for each camera view
            for t in range(v):
                R[t,t] = ensemble_vars[k][i][t]
            S[k,i-1] = np.dot(A, np.dot(Vf[k,i-1, :], A.T)) + Q
            #print(S[i-1], )
            y_minus_CAmf = y[k,i, :] - np.dot(C, np.dot(A, mf[k,i-1, :]))
           
            
            if any(part in i for i in constrained_keypoints_graph):
                # gradient terms
                grad = gradient_distance(mf[:,i,:], part, D, keypoint_ensemble_list, constrained_keypoints_graph)
                
                hess = hessian_distance(mf[:,i,:], part, D, keypoint_ensemble_list, constrained_keypoints_graph)
                # add gradient and hessian penalisaiton
                mf[k,i, :] = np.dot(A, mf[k,i-1, :]) + kalman_dot(y_minus_CAmf, S[k,i-1], C, R) + mu*grad
                
                S[k,i-1] = np.linalg.inv(np.linalg.inv(S[k,i-1])+mu*hess)
                
            else:
                mf[k,i, :] = np.dot(A, mf[k,i-1, :]) + kalman_dot(y_minus_CAmf, S[k,i-1], C, R) 
            Vf[k,i, :] = S[k,i-1] - kalman_dot(np.dot(C, S[k,i-1]), S[k,i-1], C, R)
            
    return mf, Vf, S     





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






#%%%%% FILTERING



all_mu = {0:None, 1: None}
#print(f"filtering ...")
for i in range(len(mu)):
    mfc, Vfc, Sc  = filtering_pass_with_constraint(y_obs, m0, S0, C, R, A, Q,ensemble_vars, D_ij, L, keypoint_ensemble_list, constrained_keypoints_graph=c, mu=mu[i],loss='eps')

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
    
    

    #%%%% Plots
    
    
nkeys = len(keypoint_ensemble_list)
ncams = len(camkeys)
color = ['b','g','r','c','m','y','k']
fig,ax = plt.subplots(nkeys,2*ncams,figsize=(24,12))
for n,key in enumerate(keypoint_ensemble_list):
    for j,cam in enumerate(camera_names):
        for i in range(len(mu)):
            ax[n, 2*j].plot(all_mu[i]['{}_df'.format(cam)]['ensemble-kalman_tracker'][key]["x"], color=color[i], label = ' {}'.format(mu[i]), alpha=0.3)
            ax[n,2*j].title.set_text('{}'.format(key)+'{}'.format(cam)+'_x')
            ax[n, 2*j-1].plot(all_mu[i]['{}_df'.format(cam)]['ensemble-kalman_tracker'][key]["y"], color=color[i], label = ' {}'.format(mu[i]), alpha = 0.3)
            ax[n, 2*j-1].title.set_text('{}'.format(key)+'{}'.format(cam)+'_y')
            ax[n, 2*j].legend(loc='upper right')
            ax[n,2*j-1].legend(loc='upper right')
        

#%%%V VIDEO
base_path = '/Users/clairehe/Documents/GitHub/eks/data/mirror-fish/videos-for-each-labeled-frame'
video_name = 'raw_vid'


save_file = os.path.join(base_path, f'%s_eks.mp4' % video_name)
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
