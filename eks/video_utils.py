#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 08:31:35 2023

@author: clairehe

Helper functions for video making
"""
import cv2
import os
import numpy as np
import shutil
import subprocess

def get_frames_from_idxs(cap, idxs):
    """Helper function to load video segments.
    Parameters
    ----------
    cap : cv2.VideoCapture object
    idxs : array-like
        frame indices into video
    Returns
    -------
    np.ndarray
        returned frames of shape shape (n_frames, n_channels, ypix, xpix)
    """
    is_contiguous = np.sum(np.diff(idxs)) == (len(idxs) - 1)
    n_frames = len(idxs)
    for fr, i in enumerate(idxs):
        if fr == 0 or not is_contiguous:
            cap.set(1, i)
        ret, frame = cap.read()
        if ret:
            if fr == 0:
                height, width, _ = frame.shape
                frames = np.zeros((n_frames, 1, height, width), dtype='uint8')
            frames[fr, 0, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            print(
                'warning! reached end of video; returning blank frames for remainder of ' +
                'requested indices')
            break
    return frames





def save_video(save_file, tmp_dir, framerate=20, frame_pattern='frame_%06d.jpeg'):
    """
    Parameters
    ----------
    save_file : str
        absolute path of filename (including extension)
    tmp_dir : str
        temporary directory that stores frames of video; this directory will be deleted
    framerate : float, optional
        framerate of final video
    frame_pattern : str, optional
        string pattern used for naming frames in tmp_dir
    """

    if os.path.exists(save_file):
        os.remove(save_file)

    # make mp4 from images using ffmpeg
    call_str = \
        'ffmpeg -r %f -i %s -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" %s' % (
            framerate, os.path.join(tmp_dir, 'frame_%06d.jpeg'), save_file)
    print(call_str)
    subprocess.run(['/bin/bash', '-c', call_str], check=True)

    # delete tmp directory
    shutil.rmtree(tmp_dir)


def plot_video_markers(markers_dict,keypoint_ensemble_list, camera_names, ax, frame_idx, marker_shapes, colors, alphas, mu):
    
    for j, part in enumerate(keypoint_ensemble_list):
        for c, cam in enumerate(camera_names):
            for model_id in range(len(mu)):
                ax.plot(markers_dict[model_id][cam+'_df']['ensemble-kalman_tracker'][part]['x'][frame_idx],markers_dict[model_id][cam+'_df']['ensemble-kalman_tracker'][part]['y'][frame_idx], marker_shapes[model_id], markeredgecolor='w', markersize=4, alpha=alphas[model_id], 
                        color=colors[c])
