# coding:utf-8
import tensorflow as tf
import numpy as np


def read_vgg_weights():
    vgg_path = ''
    if '.npy' in vgg_path:
        data_dict = np.load(vgg_path, encoding='latin1').item()
        for op_name in data_dict:
            # if 'fc' not in op_name:
            w, b = data_dict[op_name][0], data_dict[op_name][1]
            print(w.shape, b.shape)
            # else:


def statistic_min_side_length():
    images_dir = ''


if __name__ == '__main__':
    pass
