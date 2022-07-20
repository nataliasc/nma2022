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
# preprocess frames
################################



# %%
################################
# preprocess gaze & create saliency density map
################################
# generate nd array with gaze positions (might be null, if no data marked as -1)
gaze_pos_fr = []
for i in range(len(gaze_data)):
    positions = gaze_str_to_intlist(gaze_data[i][6:])
    gaze_pos_fr.append(positions)
gaze_pos_fr = np.array(gaze_pos_fr)

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
