"""
this file preprocesses raw data from atari gaze dataset to training data for saliency prediction

input: frames and gaze positions for each trial of subject
output: n_samples*(4_frame_episodes, 1_saliency_density_map)

"""
# load meta data to get subject num
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from preprop_util import *
from glob import glob
import os
import cv2

# fetch meta data given game name
metadata = pd.read_csv(r'raw_data/meta_data.csv')
metadata_game = metadata[metadata['GameName'] == 'breakout']
print(metadata_game.head())

# get ordered game frames and corresponding gaze data from data files
# frames resized to 84*84
trial_ids = metadata_game['trial_id'].to_numpy()
# hyperparameters controlling sampling
n_skips = 4
n_frames_per_episode = 4
n_samples = 2  # number of episodes taken from each trial
# list containing all data points
episode_samples = []
target_saliency_maps = []

for trial in trial_ids:
    print('trial id: %i' % trial)
    all_grey_frames, gaze_data = load_frames_gazetxt(trial, metadata_game)

    ################################
    # preprocess gaze & create saliency density map
    ################################
    # generate nd array with gaze maps (might be null, if no data marked as -1) per frame from gaze data
    gaze_maps = []
    n_out_of_fr = 0
    n_frame_no_gaze = 0  # count how many frames do not have any gaze data
    for i in range(len(gaze_data)):
        positions = gaze_str_to_pos(gaze_data[i][6:])
        if len(positions) > 0:  # if there is gaze data for that frame
            map_fr, out_count = gaze_pos_to_map(positions)
            gaze_maps.append(map_fr)
            n_out_of_fr += out_count
        else:
            n_frame_no_gaze += 1
            gaze_maps.append(np.zeros((210, 160)))
    gaze_maps = np.array(gaze_maps)
    print('%i gaze positions are out of frame' % n_out_of_fr)
    print('%i frames have no gaze position data' % n_frame_no_gaze)

    ################################
    # sample from frames to create (4 frame episodes, combined saliency density map)
    ################################
    for times in range(n_samples):
        start_idx = np.random.randint(0, len(all_grey_frames) - n_skips * n_frames_per_episode)
        episode = get_episode(start_idx, n_skips, n_frames_per_episode, gaze_data, all_grey_frames)
        saliency_density = create_saliency_density(start_idx, n_skips, n_frames_per_episode, [84, 84], gaze_maps)

        episode_samples.append(episode)
        target_saliency_maps.append(saliency_density)

# create tensor datasets
X = torch.tensor(episode_samples)
y = torch.tensor(target_saliency_maps)

dataset = data.TensorDataset(X, y)

print(X.shape, y.shape)
