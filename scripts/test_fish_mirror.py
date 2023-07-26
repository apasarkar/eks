#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 02:33:33 2023

@author: clairehe
"""

from eks.multiview_pca_fish import *

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




mu = [0,0.001,0.005, 0.01]
c = [('fork','mid'),('mid','chin_base')]
folder = "/eks_opti"
operator = "/20210204_Quin/"
name = "img197707"

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

#image_path = "/Users/clairehe/Documents/GitHub/eks/data/mirror-fish/labeled-data"
#im = plt.imread(image_path+operator+name+".png")
#plt.imshow(im)
#plt.suptitle("labeled "+name)


session = '20210204_Quin'
frame = 'img197707.csv'
smooth_param = 0.01
quantile_keep_pca = 50
# Get markers list from networks
markers_list = []
for model_dir in model_dirs:
    csv_file = os.path.join(model_dir, session, frame)
    df_tmp = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
    keypoint_names = [l[1] for l in df_tmp.columns[::3]]
    markers_tmp = convert_lp_dlc(df_tmp, keypoint_names, model_name=tracker_name)
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
print(y_obs.shape)
##### Set values for kalman filter #####
m0 = np.asarray([0.0, 0.0, 0.0]) # initial state: mean
S0 = np.zeros((nkeys,m0.shape[0], m0.shape[0] ))
d_t = {key: None for key in range(nkeys)}
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
