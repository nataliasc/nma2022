"""
this file contains functions used for preprocessing to keep data_preprocessing_gaze file clean
"""
import numpy as np
import pandas as pd
from glob import glob
import os
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.special import softmax


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
                filler = np.full(num_datapoints - len(currentline), -1).tolist()
                currentline += filler
                all_lines.append(currentline)

    return all_lines[1:]


# %%
def get_frame_num(filename):
    """
    extract frame number from frame path (str)
    :param filename: str of frame file path
    :return: int frame number
    """
    name = filename.split('.')[0]
    frame_num = name.split('_')[-1]

    return int(frame_num)  # -1 to convert to index


def load_frames_gazetxt(trial_id, metadataframe, data_path='raw_data/breakout/'):
    '''
    given trial id load from raw_data/breakout corresponding frames and txt file for gaze
    read frames as grey scale

    :param data_path: where data for one game is stored
    :param trial_id: label for which dir to load
    :param metadataframe: dataframe containing reference
    :return: all grey scaled game frames (np.array), list of gaze data
    '''

    trial_subject_key = str(trial_id) + '_' + metadataframe[metadataframe['trial_id'] == trial_id]['subject_id'].item()
    trial_dir = glob(os.path.join(data_path, trial_subject_key + '*'))  # contain txt file and dir for frames
    # get which path in trial dir is txt or folder
    txt_file = 0 if 'txt' in trial_dir[0] else 1
    folder = 1 - txt_file
    # read data from txt file
    gaze_data = read_txt_data(trial_dir[txt_file], 50)
    # load frames from frame fir
    frames_grey = []
    num = []
    frame_glob = glob(os.path.join(trial_dir[folder] + '/', '*.png'))

    for path in frame_glob:
        grey = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # resize img to stated in original atari paper
        grey = cv2.resize(grey, (84, 84))
        frames_grey.append(grey)
        # get frame number to sort the frames
        index = get_frame_num(path)
        num.append(index)

    # sort frames by frame number
    frames_grey = np.array(frames_grey)
    idx = np.argsort(num)
    frames_grey = frames_grey[idx]

    print(len(frames_grey), len(gaze_data))
    assert len(frames_grey) == len(gaze_data)

    return frames_grey, gaze_data


# turn gaze position from str(num.num) to list[num, num]
def gaze_str_to_intlist(gaze_pos):
    '''
    convert strings of gaze position read from txt file to list containing lists of gaze pos in format [int, int]
    :param gaze_pos: list of strings of 'num.num'
    :return: one np array of [int, int] of gaze position
    '''
    for i in range(len(gaze_pos)):
        if gaze_pos[i] == 'null\n':  # if no gaze pos is available
            continue
        elif gaze_pos[i] != -1:
            x, y = gaze_pos[i].split('.')
            pos = [int(x), int(y)]
            gaze_pos[i] = pos

    return gaze_pos


def gaze_str_to_pos(gaze_pos):
    """
    try treating odd idx values as x position and even idx values as y position
    :param gaze_pos:
    :return:
    """
    gaze_positions = []
    x_pos_indices = np.arange(0, len(gaze_pos), step=2)
    for idx in x_pos_indices:
        if gaze_pos[idx] == 'null\n':  # if no gaze pos is available
            continue
        elif gaze_pos[idx] != -1:
            x = round(float(gaze_pos[idx]))
            y = round(float(gaze_pos[idx + 1]))
            gaze_positions.append([x, y])

    return gaze_positions


# from positions to gaze maps
def gaze_pos_to_map(coor):
    """
    convert gaze position to gaze map for each frame (50ms)
    :param coor: np.array of coordinate of gaze pos
    :return: resized gaze map (84*84), num of out of screen position counts
    """
    map_ = np.zeros((210, 160))  # initalise map to original frame size
    out_of_fr_count = 0
    if coor[0] != 'null\n':  # if no gaze pos is available
        for i in range(len(coor)):
            if coor[i] != -1:
                x, y = coor[i]  # x is horizontal position, y is vertical position
                if x > 159 or y > 209:  # ignore out of screen coordinates
                    # print(coor[i])
                    out_of_fr_count += 1
                    continue
                map_[y, x] = 1

    # resized_gaze_map = cv2.resize(map, (84, 84))

    return map_, out_of_fr_count


def get_episode(skip_fr, num_frames, gaze_data_list, all_frames):
    """
    sample 4 frame episodes from all frames
    :param all_frames: nd array containing all preprocessed frames
    :param start_index: the starting frame of sampled episode
    :param skip_fr: interval at which frames are sampled
    :param num_frames: total num of frames per episode
    :param gaze_data_list: list containing gaze data info per frame, used to check whether frames are from same game
    :return: nd array of episode, num_frames*fr_h*fr_w
    """
    # get start index
    start_index = np.random.randint(0, len(all_frames) - skip_fr * num_frames)
    # get indices of sampled frames
    sampled_idx = np.arange(start_index, start_index + skip_fr * num_frames, skip_fr)
    assert len(sampled_idx) == num_frames
    episode_ = np.zeros((num_frames, all_frames[0].shape[0], all_frames[0].shape[1]))

    # check whether they are from the same game
    if gaze_data_list[start_index][1] != 'null':  # check episode_id field, if not null there are multiple episodes
        # get episode id from all sampled frames
        last_epi_id = gaze_data_list[sampled_idx[0]][1]
        for fr in range(1, num_frames):
            current_epi_id = gaze_data_list[sampled_idx[fr]][1]
            if last_epi_id != current_epi_id:
                episode_, start_index = get_episode(skip_fr, num_frames, gaze_data_list, all_frames)
            else:
                episode_ = all_frames[sampled_idx]
    else:
        episode_ = all_frames[sampled_idx]

    return episode_, start_index


def create_saliency_density(start_index, skips, num_frames, density_map_dim, all_gaze_maps):
    """
    compile the saliency density for corresponding episodes
    :param start_index: index where episode starts
    :param skips: num of skip frames
    :param num_frames: total num of frames in episode
    :param density_map_dim: the dimensions of desirable density map, for resizing
    :param all_gaze_maps: nd array containing all processed gaze maps
    :return: one saliency density map
    """
    saliency_density_map = np.zeros(all_gaze_maps[0].shape)
    map_count = skips * num_frames  # total number of gaze position maps to be combined
    # compile all included gaze position maps
    for m in range(map_count):
        saliency_density_map += all_gaze_maps[start_index + m]

    # resize
    saliency_density_map = cv2.resize(saliency_density_map, (84, 84))
    # gaussian smoothing
    saliency_density_map = gaussian_filter(saliency_density_map, sigma=5)
    # softmax
    saliency_density_map = softmax(saliency_density_map)

    return saliency_density_map
