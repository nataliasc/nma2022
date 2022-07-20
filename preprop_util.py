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
                filler = np.full(num_datapoints - len(currentline), np.NaN).tolist()
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
        grey = cv2.resize(grey, (110, 84))
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


def crop_frame(frames, w, h):
    """
    crop the center w*h of the image
    :param frames: 3d np array containing all frames
    :param w: cropped width
    :param h: cropped height
    :return: 3d np array with cropped frames
    """
    _, x, y = frames.shape
    startx = int(x/2 - w/2)
    starty = int(y/2 - h/2)

    return frames[:, startx:startx+w, starty:starty+h]