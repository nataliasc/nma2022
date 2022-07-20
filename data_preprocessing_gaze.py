"""
this file preprocesses raw data from atari gaze dataset to training data for saliency prediction

input: frames and gaze positions for subjects
output: n_samples*(4_frame_episodes, 1_saliency_density_map)

"""
# load meta data to get subject num
import numpy as np
import pandas as pd
from glob import glob
import os
import cv2

metadata = pd.read_csv(r'raw_data/meta_data.csv')
metadata_breakout = metadata[metadata['GameName'] == 'breakout']
print(metadata_breakout.head())


# %%
# load frame dir and gaze txt for subject
def read_txt_data(txt_path, time_frame):
    '''
    read data from txt file containing gaze pos
    :param txt_path: file path to txt file
    :param time_frame: how much ms of gaze pos to get from file, per 50ms is 100 gaze positions
    :return: list of lists containing data values per line  shape: n_frame*time_frame
    '''
    with open(txt_path, 'r') as filestream:
        num_datapoints = time_frame * 2 + 6  # first couple of colums plus taken gaze pos
        all_lines = []
        for line in filestream:
            currentline = line.split(",")
            if len(currentline) > num_datapoints:
                all_lines.append(currentline[:num_datapoints])
            else:
                filler = np.full(num_datapoints - len(currentline), np.NaN).tolist()
                currentline += filler
                all_lines.append(currentline)

    return all_lines[1:]


def load_frames_gazetxt(trial_id, metadataframe, data_path='raw_data/breakout/'):
    '''
    given trial id load from raw_data/breakout corresponding frames and txt file for gaze
    read frames as grey scale

    :param data_path: where data for one game is stored
    :param trial_id: label for which dir to load
    :param metadataframe: dataframe containing reference
    :return: all grey scaled game frames, list of gaze data
    '''

    trial_subject_key = str(trial_id) + '_' + metadataframe[metadataframe['trial_id'] == trial_id]['subject_id'].item()
    trial_dir = glob(os.path.join(data_path, trial_subject_key + '*'))  # contain txt file and dir for frames
    # read data from txt file
    gaze_data = read_txt_data(trial_dir[0], 50)
    # load frames from frame fir
    frames_grey = []
    frame_glob = glob(os.path.join(trial_dir[1] + '/', '*.png'))
    for path in frame_glob:
        grey = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        frames_grey.append(grey)

    assert len(frames_grey) == len(gaze_data)

    return frames_grey, gaze_data


all_grey_frames, gaze_data = load_frames_gazetxt(58, metadata_breakout)

# %%
# get data from txt file
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
