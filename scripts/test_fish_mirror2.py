#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 02:33:33 2023

@author: clairehe

Script to run over fish datasets 

"""
import cv2

from eks.multiview_pca_fish2 import *
from eks.video_utils import *
from collections import defaultdict


camera_names = ['main', 'top']
keypoint_ensemble_list = ['anal','caudal_d','caudal_v','fork','tail_neck', 'stripeA', 'stripeP']
  #['mid', 'fork', 'chin_base']# ,'fork','chin_base']
constraint = [('fork','caudal_v'),('fork','caudal_d')]
nkeys = len(keypoint_ensemble_list)
#keypoint_ensemble_list = [ 'head', 'chin_base', 'chin1_4', 'chin_half','chin3_4', 'chin_tip', 'mid', 'fork',
# 'stripeA', 'stripeP', 'tail_neck', 'dorsal', 'anal', 'caudal_d', 'caudal_v']
tracker_name = 'heatmap_mhcrnn_tracker'
num_cameras = len(camera_names)
ncams = num_cameras
labeled_data = pd.read_csv("/Users/clairehe/Documents/GitHub/eks/data/mirror-fish/CollectedData_new.csv", header = [1,2])



mu = [0, 0.01]



session = '/20210202_Sean' #'20210204_Quin'
folder = "/eks_opti"
operator = "/20210202_Sean/" #"/20210204_Quin/"
# name ='img023203' # "img048416" 
#name =  "img197707" 
frame = session + '.csv' # name+'.csv'

#baseline = pd.read_csv("/Users/clairehe/Documents/GitHub/eks/data/misc/mirror-fish_ensemble-predictions/eks"+operator+name+".csv", header=[ 1, 2],index_col=0)
#new = pd.read_csv("/Users/clairehe/Documents/GitHub/eks/data/misc/one-video-mirror-fish-predictions"+folder+operator+name, header=[ 1, 2], index_col=0)
#baseline0 = pd.read_csv("/Users/clairehe/Documents/GitHub/eks/data/misc/mirror-fish_ensemble-predictions/eks"+operator+name+".csv", header=[0, 1, 2],index_col=0)


# NOTE! replace this path with an absolute path where you want to save EKS outputs
eks_save_dir = '/Users/clairehe/Documents/GitHub/eks/data/misc/one-video-mirror-fish-predictions/eks_opti/'

# path for prediction csvs
file_path = '/Users/clairehe/Downloads/mirror-fish_ensemble-predictions_long' # long videos
# '/Users/clairehe/Documents/GitHub/eks/data/misc/mirror-fish_ensemble-predictions' shrot videos

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
# labeled_data.columns = ['_'.join(tup).rstrip('_') for tup in labeled_data.columns.values]

# img_id = labeled_data.loc[['labeled-data'+operator+name+'.png' in s for s in labeled_data.bodyparts_coords]].index[0]

n_latent=3
#%% set distance constraint projection for a few q's in keypoint_ensemble_list



mask = []
for keys in keypoint_ensemble_list:
    for cam in camkeys:
#        for coord in ["_x", "_y"]:
        mask.append(keys+cam)
            


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
q = np.empty((n, num_mar, n_latent))


for j, keypoint_ensemble in enumerate(keypoint_ensemble_list):
    markers_list_cameras = [[] for i in range(num_cameras)]
    for i,cam in enumerate(camkeys):
        tmp = []

        for m in markers_list.keys():
            if cam in m[0] and keypoint_ensemble in m[0]:
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
    labeled_pca, labeled_var = pca(y, n_latent)
    
    q[j,:,:] = labeled_pca.transform(y)
    y_obs[j,:,:] = y

  # Define the size of L
L_initial = np.tril(np.eye(n_latent)).flatten()
L0 = find_linear_transformation(q, L_initial)

D = get_3d_distance(q, L0, keypoint_ensemble_list, constraint)



L = np.zeros((nkeys*n_latent, nkeys*n_latent))
for i in range(nkeys):
    L[i*(n_latent):(i+1)*n_latent,i*(n_latent):(i+1)*n_latent ] = L0
#%% REPROCESSING



# Get markers list from networks
markers_list = []
for model_dir in model_dirs:
    csv_file = model_dir+frame
    print(csv_file)
    # os.path.join(model_dir, session, frame) # short ivdeos
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
markers_all = []
for n, keypoint_ensemble in enumerate(keypoint_ensemble_list):
    markers_list_cameras = [[] for i in range(ncams)]
    for m in markers_list:
        for camera in range(ncams):
            markers_list_cameras[camera].append(
                m[[key for key in m.keys() 
                   if camera_names[camera] in key 
                   and 'likelihood' not in key 
                   and keypoint_ensemble in key]
                 ]
                
            )
    # ENSEMBLING PER KEYPOINTS
    # keypoint, #camera, #number of network, # (51, 2)
    markers_all.append(markers_list_cameras)
    scaled_ensemble_preds, good_frames, good_scaled_ensemble_preds,ensemble_vars,means_camera = ensembling_multiview(markers_list_cameras, keypoint_ensemble, smooth_param, quantile_keep_pca, camera_names)
    scaled_dict.append(scaled_ensemble_preds)
    good_frames_dict.append(good_frames)
    good_preds_dict.append(good_scaled_ensemble_preds)
    ensemble_vars_dict.append(ensemble_vars)
    means_camera_dict.append(means_camera)
ensemble_vars = np.array(ensemble_vars_dict)

T = scaled_dict[0].shape[0]

stacked_preds,ensemble_pca,ensemble_ex_var,ensemble_pcs,good_ensemble_pcs =  multiview_pca_bodyparts(scaled_dict,good_preds_dict,good_frames_dict)

# scaled_dict = np.array(scaled_dict).reshape((T, nkeys*2*ncams))
# good_preds_dict = np.array(good_preds_dict).reshape((len(good_preds_dict[0]),nkeys*2*ncams))
# good_ensemble_pcs= {key: None for key in range(len(good_frames_dict))}
# # PCA

# ensemble_pca, ensemble_ex_var = pca(good_preds_dict, n_latent*nkeys)
# ensemble_pcs = ensemble_pca.transform(scaled_dict)


# for key in range(len(good_frames_dict)):
#     good_ensemble_pcs[key] = ensemble_pcs[good_frames_dict[key],3*key:3*(key+1)]

# return scaled_dict,ensemble_pca,ensemble_ex_var,ensemble_pcs,good_ensemble_pcs

# y_obs = scaled_dict

#compute center of mass
#latent variables (observed)
good_z_t_obs = good_ensemble_pcs #latent variables - true 3D pca

    

good_z_t_obs = np.array(list(good_z_t_obs.values())).reshape((len(good_z_t_obs[0]),nkeys*n_latent))

var_good_obs = np.var(good_z_t_obs, axis= 0)


#print(y_obs.shape)
##### Set values for kalman filter #####
m0 = np.zeros(nkeys*n_latent) # initial state: mean
S0 = np.zeros((m0.shape[0], m0.shape[0] ))
# need different variance for each bodyparts 
np.fill_diagonal(S0, var_good_obs)



d_t = good_z_t_obs[1:,]-good_z_t_obs[:-1,]

Q = smooth_param*np.cov(d_t.T)

A = np.eye(nkeys*n_latent) #state-transition matrix,
# Q = np.asarray([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]) #state covariance matrix?????


# C = ensemble_pca.components_.T # Measurement function is inverse transform of PCA
# R = np.eye(ensemble_pca.components_.shape[1]) # placeholder diagonal matrix for ensemble variance


# assume y is shaped as y[0,:] [bodypart, camera, coord] 6 [bodypart, camera, coord] 6 ... ex shape (51,18)
y_obs = np.swapaxes(np.array(stacked_preds),0,1).reshape(T, 2*ncams*nkeys)



# tmp = []
# for i in range(nkeys):
#      tmp.append(np.repeat([ensemble_pca.components_[i,:]],nkeys, axis=0))
#      H = np.vstack(tmp)
# C = np.repeat(H, nkeys, axis=1).T
# C = ensemble_pca.components_.T

R = np.eye(2*ncams*nkeys)
ens_var= ensemble_vars.reshape((T,2*ncams*nkeys))

C= np.zeros((2*ncams*nkeys,n_latent*nkeys))
for i in range(nkeys):
    C[i*2*ncams:(i+1)*2*ncams,i*n_latent:(i+1)*n_latent]=ensemble_pca.components_.T
    
D_ij =  np.mean(D, axis=0)
eps = np.var(D, axis=0)




#%%%%% FILTERING


from eks.ensemble_kalman import filtering_pass, smooth_backward
from eks.multiview_pca_fish2 import filtering_pass_with_constraint
all_mu = {0:None}
tract_dict = {0:None}
for i,mu0 in enumerate(mu):
    print(f"filtering ...")
    mfc, Vfc, Sc, tract_dict[i]  = filtering_pass_with_constraint(y_obs, m0, S0, C, R, A, Q,ens_var, D_ij,L, keypoint_ensemble_list, constrained_keypoints_graph=constraint, mu=mu0,loss='squared')
    #mfc, Vfc, Sc  = filtering_pass(y_obs, m0, S0, C, R, A, Q, ens_var)
    #filtering_pass_with_constraint(y_obs, m0, S0, C, R, A, Q,ensemble_vars, D_ij, L, keypoint_ensemble_list, constrained_keypoints_graph=constraint, mu=mu[i],loss='eps')
    
    ## Do the smoothing step
    #print("done filtering")
    
    
    
    y_m_filt = np.dot(C, mfc.T).T
    y_v_filt = np.swapaxes(np.dot(C, np.dot(Vfc, C.T)), 0, 1)
    #print(f"smoothing {keypoint_ensemble_list[k]}...")
    ms, Vs, _ = smooth_backward(y_obs, mfc, Vfc, Sc, A, Q, C)
    #print("done smoothing")k
    
    # Smoothed posterior over yb
    y_m_smooth = np.dot(C, ms.T).T
    y_v_smooth = np.swapaxes(np.dot(C, np.dot(Vs, C.T)), 0, 1)
    
    ncams = len(camkeys)
    camera_dfs = {}
    camera_indices = []
    for camera in range(ncams):
        camera_indices.append([camera*2, camera*2+1])
    
    
            
    for camera, camera_name in enumerate(camera_names):
        key_df = []
        for k, keypoint_ensemble in enumerate(keypoint_ensemble_list):
            
            xyl_labels = ["x", "y", "var_x", "var_y"]
            pdindex = pd.MultiIndex.from_product(
                [[keypoint_ensemble], xyl_labels],
                names=["bodyparts", "coords"],
            )
            var_x = np.empty(y_m_smooth.T[k*2*ncams+camera_indices[camera][0]].shape)
            var_x[:] = y_v_smooth[:,k*ncams,k*ncams]
            var_y = np.empty(y_m_smooth.T[k*2*ncams+camera_indices[camera][0]].shape)
            var_y[:] = y_v_smooth[:,k*ncams+1,k*ncams+1]
            # print(keypoint_ensemble+'_x, _y', k*6+camera_indices[camera][0],k+camera_indices[camera][1])
            pred_arr = pd.DataFrame(np.vstack([
                y_m_smooth.T[k*2*ncams +camera_indices[camera][0]] + means_camera_dict[k][camera_indices[camera][0]],
                y_m_smooth.T[k*2*ncams +camera_indices[camera][1]] + means_camera_dict[k][camera_indices[camera][1]],
                var_x, var_y
            ]).T, columns = pdindex)
            key_df.append(pred_arr)
        
        #print(means_camera[camera_indices[camera][0]],means_camera[camera_indices[camera][1]])
        camera_dfs[camera_name + '_df'] = pd.concat(key_df,axis=1)
    all_mu[i] = camera_dfs
        

    #%%%% Plots the traces
    
    
nkeys = len(keypoint_ensemble_list)
ncams = len(camkeys)
color = ['b','g','r','c','m','y','k']

for n,key in enumerate(keypoint_ensemble_list):
    fig,ax = plt.subplots(2*ncams,1,figsize=(40,40))
    for j,cam in enumerate(camera_names):
        for i in range(len(mu)):
            
            ax[2*j].plot(all_mu[i]['{}_df'.format(cam)][key]["y"], color=color[i], label = ' {}'.format(mu[i]), alpha=0.5,linestyle='-.')
            ax[2*j].title.set_text('{}'.format(key)+'{}'.format(cam)+'_x')
            ax[2*j-1].plot(all_mu[i]['{}_df'.format(cam)][key]["x"], color=color[i], label = ' {}'.format(mu[i]), alpha = 0.5,linestyle='-.')
           
        # add network predictions
        for p in range(5):
            ax[2*j-1].plot(markers_all[n][j][p][key+'_'+cam+'_x'], color='grey', label = ' network {}'.format(p), alpha=0.2, linestyle='dotted')
            ax[2*j].plot(markers_all[n][j][p][key+'_'+cam+'_y'], color='grey', label = ' network {}'.format(p), alpha=0.2, linestyle='dotted')
        ax[2*j].title.set_text('{}'.format(key)+'{}'.format(cam)+'_y')
        ax[1].legend(loc='lower right')
        ax[2].legend(loc='lower right')
    fig.suptitle('Trace plots with constraints {}'.format(constraint))
    fig.savefig('/Users/clairehe/Documents/GitHub/paninski-lab/figures/trace_{}.png'.format(key))
    
    
    
    
#%%%% Plot the limb length
color = ['r','g']
style = ['-.','dotted']
fig, ax = plt.subplots(1,1,figsize=(20,10))
for i in range(len(tract_dict)):
    for j in range(tract_dict[0].shape[1]):
        ax.plot(tract_dict[i][1:,j], color = color[i], linestyle = style[j], label='limb length edge {}'.format(constraint[j]) +' with mu {}'.format(mu[i]))
ax.legend()
plt.title('Limb length trace per frame with constraint mu {}'.format(mu[1])+' against no constraint')

#%%%V VIDEO FOR ONE MU
base_path = '/Users/clairehe/Downloads/mirror-fish_ensemble-predictions_long' # longn videos
# short videos '/Users/clairehe/Documents/GitHub/eks/data/mirror-fish/videos-for-each-labeled-frame'
video_name = 'raw_vid'

name= session[1:]
save_file = os.path.join(base_path, f'%s_eks' % video_name + '_mu_{}.mp4'.format(mu[0]))
framerate=1

cap = cv2.VideoCapture(base_path+operator+name+'.mp4')

marker_shapes = ["o",'s'] #marker_shapes = ['D','o', '^', 's','v'] # 4 models 
colors = ['red', 'green', 'blue'] # 3 views

single_alpha = .5
single_alpha_list = [single_alpha]*len(all_mu)
alphas = [1.0]+single_alpha_list 
model_labels = ['ensemble', f'kalman, E:{mu}']
frame_idxs = [i for i in range(len(all_mu[0]['main_df']['mid']['x']))]
tmp_dir = os.path.join(os.path.dirname(save_file), 'tmpZzZ2')
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
for idx in frame_idxs: #[250:700]
    # important!! otherwise each frame will plot on top of the last
    ax.clear()

    frame = get_frames_from_idxs(cap, [idx])
    # plot original frame
    ax.imshow(frame[0, 0], vmin=0, vmax=255, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])

    patches = []
    for i,key in enumerate(keypoint_ensemble_list): 
        for j, cam in enumerate(camkeys):
            for p in range(5):
                ax.plot(markers_all[i][j][p][key+cam+'_x'][idx],markers_all[i][j][p][key+cam+'_y'][idx], 'v', markeredgecolor='w', markersize=4, alpha=0.5, 
                        color=colors[0]) 
    
    plot_video_markers(all_mu, keypoint_ensemble_list, camera_names, ax, idx, marker_shapes, colors[1], alphas, mu)
    
    plt.legend(mu)
    plt.title(constraint)
    im = ax.text(0.02, 0.98, 'frame %i' % idx, **txt_fr_kwargs)
    plt.savefig(os.path.join(tmp_dir, 'frame_%06i.jpeg' % idx))
save_video(save_file, tmp_dir, framerate, frame_pattern='frame_%06i.jpeg')

#%%% SLICED VIDEO


import matplotlib.patches as mpatches
base_path = '/Users/clairehe/Downloads/mirror-fish_ensemble-predictions_long' # longn videos
# short videos '/Users/clairehe/Documents/GitHub/eks/data/mirror-fish/videos-for-each-labeled-frame'
video_name = 'raw_vid'
cstr_name = 'fork_caudal'
save_file = os.path.join(base_path, f'%s_eks' % video_name +'_mu_{}'.format(mu[1])+ '_constraint_{}.mp4'.format(cstr_name))
slice_l = [1100,1700]

name= session[1:]
framerate=1


def sliced_video_from_eks(all_mu, y_v_smooth, slice_l, base_path, video_name, save_file, name, framerate=1):

    fig, ax = plt.subplots(1,1,figsize=(20, 20))
    txt_fr_kwargs = {
        'fontsize': 14, 'color': [1, 1, 1], 'horizontalalignment': 'left',
        'verticalalignment': 'top', 'fontname': 'monospace',
        'bbox': dict(facecolor='k', alpha=0.25, edgecolor='none'),
        'transform': ax.transAxes
    }
    
    
    cap = cv2.VideoCapture(base_path+operator+name+'.mp4')
    
    marker_shapes = ['o', '^', 'D', 's'] # 4 models 
    model_colors = ['r', 'g'] # bodyparts
    colors= ['red'] + ['green']*(len(all_mu[0])-1)
    single_alpha = .4
    single_alpha_list = [single_alpha]*len(all_mu)
    alphas = [0.8]+single_alpha_list 
    model_labels = ['no constraint', f'constraints:{mu}']
    frame_idxs = [i for i in range(len(all_mu[0]['main_df'][keypoint_ensemble_list[0]]['x']))]
    tmp_dir = os.path.join(os.path.dirname(save_file), 'tmpZzZ2')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    # plot 
    
    import tqdm 
    for idx in tqdm.tqdm(frame_idxs[slice_l[0]:slice_l[1]]):
        # important!! otherwise each frame will plot on top of the last
        ax.clear()
    
        frame = get_frames_from_idxs(cap, [idx])
        # plot original frame
        ax.imshow(frame[0, 0], vmin=0, vmax=255, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    
        # patches = []
        # for i,key in enumerate(keypoint_ensemble_list): 
        #     for j, cam in enumerate(camkeys):
        #         for p in range(5):
        #             ax.plot(markers_all[i][j][p][key+cam+'_x'][idx],markers_all[i][j][p][key+cam+'_y'][idx], 'v', markeredgecolor='w', markersize=10, alpha=0.5, 
        #                     color=colors[0]) 
        
       # plot_video_markers(all_mu, keypoint_ensemble_list, camera_names, ax, idx, marker_shapes, colors, alphas, mu)
        
        plot_variance_markers(all_mu, y_v_smooth,keypoint_ensemble_list, camera_names, ax, idx, marker_shapes, colors, alphas, mu)
    
    
        plt.legend(mu)
        
        #print(idx)
        im = ax.text(0.02, 0.98, 'frame %i' % idx, **txt_fr_kwargs)
        plt.title(constraint)
        plt.savefig(os.path.join(tmp_dir, 'frame_%06i.jpeg' % (idx-slice_l[0])))
    save_video(save_file, tmp_dir, framerate, frame_pattern='frame_%06i.jpeg')
    





sliced_video_from_eks(slice_l, base_path, video_name, save_file, name, framerate=1)
