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
    num_classes = 4
    tau = 0.9

    inputs = tf.placeholder(dtype=tf.float32, shape=[1, 3, 3, num_classes])
    label_batch = tf.placeholder(dtype=tf.int32, shape=[1, 3, 3, 1])

    score_map = tf.nn.softmax(inputs, dim=-1)
    score_map_max = tf.reduce_max(score_map, axis=3, keep_dims=False)
    # y_il = tf.maximum(score_map_max, tf.constant(tau, tf.float32, label_batch.get_shape()[:-1]))
    y_il = tf.maximum(score_map_max, tf.fill(tf.shape(label_batch)[:-1], tau))
    _s_il = 1.0 - score_map_max
    _y_il = 1.0 - y_il
    a = tf.expand_dims(tf.div(_y_il, _s_il), axis=3)
    y_ic = tf.concat([a for i in range(num_classes)], axis=3)
    y_ic = tf.multiply(score_map, y_ic)
    b = tf.expand_dims(y_il, axis=3)
    y_il_ = tf.concat([b for i in range(num_classes)], axis=3)
    lab_hot = tf.squeeze(tf.one_hot(label_batch, num_classes, dtype=tf.float32), axis=3)
    gt_batch = tf.where(tf.equal(lab_hot, 1.), y_il_, y_ic)
    gt_batch2 = tf.clip_by_value(gt_batch, 0., 1.)

    with tf.Session() as sess:
        feed_dict = {inputs: np.random.rand(1, 3, 3, num_classes).astype(np.float32),
                     label_batch: np.array([[[[0], [1], [2]], [[3], [1], [2]], [[0], [1], [3]]]])}
        c, d, e, f, g, h, i, j, k = sess.run(
            [score_map, score_map_max, y_il, _s_il, _y_il, y_ic, lab_hot, gt_batch, gt_batch2], feed_dict)
        print(c)
        print('c-----------------')
        print(d)
        print('d-----------------')
        print(e)
        print('e-----------------')
        print(f)
        print('f-----------------')
        print(g)
        print('g-----------------')
        print(h)
        print('h-----------------')
        print(i)
        print('i-----------------')
        print(j)
        print('j-----------------')
        print(np.sum(k, axis=3))
