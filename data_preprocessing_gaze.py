"""
this file preprocesses raw data from atari gaze dataset to training data for saliency prediction

input: frames and gaze positions for subjects
output: n_samples*(4_frame_episodes, 1_saliency_density_map)

"""
# load meta data to get subject num
import numpy as np
import pandas as pd
from preprop_util import *
from glob import glob
import os
import cv2
import matplotlib.pyplot as plt

metadata = pd.read_csv(r'raw_data/meta_data.csv')
metadata_breakout = metadata[metadata['GameName'] == 'breakout']
print(metadata_breakout.head())

# %%
# get ordered game frames and corresponding gaze data from data files
all_grey_frames, gaze_data = load_frames_gazetxt(58, metadata_breakout)

# %%
################################
# preprocess gaze & create saliency density map
################################
# generate nd array with gaze maps (might be null, if no data marked as -1) per frame from gaze data
# gaze_maps = []
# n_out_of_fr = 0
# for i in range(len(gaze_data)):
#     positions = gaze_str_to_intlist(gaze_data[i][6:])
#     map_fr, out_count = gaze_pos_to_map(positions)  # TODO debug gaze pos outside of frame
#     gaze_maps.append(map_fr)
#     n_out_of_fr += out_count
# gaze_maps = np.array(gaze_maps)
# print('%i gaze positions are out of frame' % n_out_of_fr)

# %%
gaze_maps = []
n_out_of_fr = 0
n_frame_no_gaze = 0  # count how many frames do not have any gaze data
for i in range(len(gaze_data)):
    positions = gaze_str_to_pos(gaze_data[i][6:])
    if len(positions) > 0:  # if there is gaze data for that frame
        map_fr, out_count = gaze_pos_to_map(positions)  # TODO debug gaze pos outside of frame
        gaze_maps.append(map_fr)
        n_out_of_fr += out_count
    else:
        n_frame_no_gaze += 1
        gaze_maps.append(np.zeros((210, 160)))
gaze_maps = np.array(gaze_maps)
print('%i gaze positions are out of frame' % n_out_of_fr)
print('%i frames have no gaze position data' % n_frame_no_gaze)

# %%
combined_map = np.sum(gaze_maps, axis=0)
plt.imshow(combined_map)
plt.show()

# %%
################################
# sample from frames to create (4 frame episodes, combined saliency density map)
################################
# hyperparameters controlling sampling
n_skips = 4
n_frames_per_episode = 4
n_samples = 1
data_set = []


# for nsamples of times
#     random start frame
#     get_episode(start_idx, skip, nframes) (check if the frames are from the same game, if not resample)
#
#     compile saliency density map (stack, resize)
#     create tuple (episode, targetdensity) of nd arrays
#     append to data_set




for times in range(n_samples):
    start_idx = np.random.randint(0, len(all_grey_frames))
    episode = get_episode(start_idx, n_skips, n_frames_per_episode, gaze_data, all_grey_frames)

# %%

# %%
# per subject
# load all frames from folder as grey scale
# separate episodes of games
# nSkip, nFramePerEp

# create list of episodes
# input all frames per play episode
# output n_sampled_4frameep*4*pixelw*pixelh
# can sample multiple times at different starting points

# crop output from previous section
# output n_sampled_4frameep*4*84*84


# gaze prediction
# load from txt gaze positions from one subject
# per frame crop 50ms of gaze position
# compile for each nskip*nframeperep a gaze map with binary values
# gaussian smoothing
# pass through softmax
# output n_sampled_4frameep*4*pixelw*pixelh saliency density map
# crop to n_sampled_4frameep*4*84*84
#
# zip output from first and second secontion into tensors of n_samples*(4_frame_episodes, 1_saliency_density_map)
#
# save for subject

# if __name__=main:
#     for all subjects
#         do above
