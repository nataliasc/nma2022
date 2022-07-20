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
# crop all grey frames
img_w = 84
img_h = 84

cropped_frames = crop_frame(all_grey_frames, img_w, img_h)

# %%
for i in range(10):
    plt.imshow(cropped_frames[i])
    plt.show()
# %%
# turn gaze position from str(num.num) to list[num, num]
def gaze_str_to_intlist(gaze_pos):
    '''
    convert strings of gaze position read from txt file to list containing lists of gaze pos in format [int, int]
    :param gaze_pos: list of strings of 'num.num'
    :return: one list of [int, int] of gaze position
    '''
    for i in range(len(gaze_pos)):
        if gaze_pos[i] != np.NaN:
            x, y = gaze_pos[i].split('.')
            pos = [int(x), int(y)]
            gaze_pos[i] = pos

    return gaze_pos

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
