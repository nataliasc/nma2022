"""
this file contains functions used for preprocessing to keep data_preprocessing_gaze file clean
"""
import numpy as np
import pandas as pd
from glob import glob
import os
import cv2
import matplotlib.pyplot as plt


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
    # read data from txt file
    gaze_data = read_txt_data(trial_dir[0], 50)
    # load frames from frame fir
    frames_grey = []
    num = []
    frame_glob = glob(os.path.join(trial_dir[1] + '/', '*.png'))

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

    assert len(frames_grey) == len(gaze_data)

    return frames_grey, gaze_data


# turn gaze position from str(num.num) to list[num, num]
def gaze_str_to_intlist(gaze_pos):
    '''
    convert strings of gaze position read from txt file to list containing lists of gaze pos in format [int, int]
    :param gaze_pos: list of strings of 'num.num'
    :return: one list of [int, int] of gaze position
    '''
    for i in range(len(gaze_pos)):
        if gaze_pos[i] == 'null\n':  # if no gaze pos is available
            continue
        elif gaze_pos[i] != -1:
            x, y = gaze_pos[i].split('.')
            pos = [int(x), int(y)]
            gaze_pos[i] = pos

    return gaze_pos
