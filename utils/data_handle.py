# coding: utf-8
import os
import re

import tensorflow as tf
import numpy as np


def load_weight(weight_path, saver, sess):
    if '.npy' in weight_path:
        data_dict = np.load(weight_path, encoding='latin1').item()
        for op_name in data_dict:
            if 'fc' in op_name: continue
            w, b = data_dict[op_name][0], data_dict[op_name][1]
            # print('discriminator_add_vgg/image_' + op_name + '/weights')
            # # tf.global_variables()
            # d_var = [v for v in tf.trainable_variables() if 'discriminator' in v.name]
            # for item in d_var:
            #     print(item.op.name)
            w_var = [v for v in saver if op_name in v.op.name and 'weights' in v.op.name][0]
            b_var = [v for v in saver if op_name in v.op.name and 'biases' in v.op.name][0]
            sess.run(w_var.assign(w))
            sess.run(b_var.assign(b))
            # except ValueError:
            #     print("fail to load %s layer in vgg16.npy!" % op_name)
    else:
        try:
            cp_path = tf.train.latest_checkpoint(weight_path)
            print("load path: %s" % cp_path)
            saver.restore(sess, cp_path)
            print("already loaded the prior model, go on training.")
            return int(re.search(r'-(.*)', cp_path).group(1))
        except:
            print("fail to load model weight!")
            return
    return


def save_weight(weight_path, saver, sess, global_step):
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    saver.save(sess, weight_path, global_step)


def generate_list():
    filepath = '/data/rui.wu/GZHermit/Workspace/SegModels/dataset/train.txt'
    dataurl = '/data/bo718.wang/zhaowei/data/516data/VOC2012trainval/VOC2012'
    # if os.path.exists(filepath):
    #     os.remove(filepath)
    hehe = os.listdir(dataurl + '/JPEGImages')
    namelist = []
    print(len(hehe))
    for h in hehe:
        namelist.append("/JPEGImages/" + h + " /SegmentationClass/" + h.strip('.jpg') + ".png\n")
    assert len(hehe) == len(namelist)
    with open(filepath, 'w') as f:
        f.writelines(namelist)


if __name__ == '__main__':
    generate_list()
