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


def check():
    num_classes = 4
    tau = 0.9

    inputs = tf.placeholder(dtype=tf.float32, shape=[1, 3, 3, num_classes])
    label_batch = tf.placeholder(dtype=tf.int32, shape=[1, 3, 3, 1])

    score_map = tf.nn.softmax(inputs, dim=-1)
    score_map_sum = tf.reduce_sum(score_map, axis=3, keep_dims=False)
    score_map_max = tf.reduce_max(score_map, axis=3, keep_dims=False)
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
    c = tf.expand_dims(tf.reduce_mean(gt_batch2, axis=3), axis=3)
    nor_sum = tf.concat([c for i in range(num_classes)], axis=3)
    gt_batch3 = gt_batch2 / nor_sum

    with tf.Session() as sess:
        feed_dict = {inputs: np.random.rand(1, 3, 3, num_classes).astype(np.float32),
                     label_batch: np.array([[[[0], [1], [2]], [[3], [1], [2]], [[0], [1], [3]]]])}
        c, d, d_, e, f, g, h, i, j, k, l = sess.run(
            [score_map, score_map_sum, score_map_max, y_il, _s_il, _y_il, y_ic, lab_hot, gt_batch, gt_batch2,
             gt_batch3], feed_dict)
        print(c)
        print('score_map-----------------')
        print(d)
        print('score_map_sum-----------------')
        print(d_)
        print('score_map_max-----------------')
        print(e)
        print('y_il-----------------')
        print(f)
        print('_s_il-----------------')
        print(g)
        print('_y_il-----------------')
        print(h)
        print('y_ic-----------------')
        print(i)
        print('lab_hot-----------------')
        print(j)
        print('gt_batch-----------------')
        print(k)
        print('gt_batch2-----------------')
        print(l)
        print('gt_batch3-----------------')
        print(np.sum(k, axis=3))


def check2():
    num_classes = 21
    tau = 0.9

    inputs = tf.placeholder(dtype=tf.float32, shape=[1, 3, 3, num_classes])
    label_batch = tf.placeholder(dtype=tf.int32, shape=[1, 3, 3, 1])
    lab_hot = tf.squeeze(tf.one_hot(label_batch, num_classes, dtype=tf.float32), axis=3)

    score_map = tf.nn.softmax(inputs, dim=-1)
    score_map_max = tf.reduce_max(score_map, axis=3, keep_dims=True)
    score_map_max = tf.maximum(score_map_max, tf.fill(tf.shape(score_map_max), tau))
    score_map_maxs = tf.concat([score_map_max for i in range(num_classes)], axis=3)
    gt_batch = tf.where(tf.equal(lab_hot, 1.), score_map_maxs, score_map)
    y_il = 1. - score_map_maxs
    s_il = 1. - score_map
    y_ic = tf.multiply(score_map, tf.div(y_il, s_il))
    gt_batch = tf.where(tf.equal(lab_hot, 0.), y_ic, gt_batch)
    sums = tf.reduce_sum(gt_batch, axis=3)
    temp = tf.expand_dims((sums - tf.ones_like(sums, dtype=tf.float32)) / num_classes, axis=3)
    gt_batch = gt_batch - tf.concat([temp for i in range(num_classes)], axis=3)
    sums = tf.reduce_sum(gt_batch, axis=3)
    # gt_batch = tf.nn.softmax(gt_batch)

    with tf.Session() as sess:
        feed_dict = {inputs: np.random.rand(1, 3, 3, num_classes).astype(np.float32),
                     label_batch: np.array([[[[0], [1], [2]], [[3], [1], [2]], [[0], [1], [3]]]])}
        c, d = sess.run([gt_batch, sums], feed_dict)
        print(c)
        print(d)
        # c, d, d_, e, f, g, h, i, j, k, l = sess.run(
        #     [score_map, score_map_sum, score_map_max, y_il, _s_il, _y_il, y_ic, lab_hot, gt_batch, gt_batch2,
        #      gt_batch3], feed_dict)
        # print(c)
        # print('score_map-----------------')
        # print(d)
        # print('score_map_sum-----------------')
        # print(d_)
        # print('score_map_max-----------------')
        # print(e)
        # print('y_il-----------------')
        # print(f)
        # print('_s_il-----------------')
        # print(g)
        # print('_y_il-----------------')
        # print(h)
        # print('y_ic-----------------')
        # print(i)
        # print('lab_hot-----------------')
        # print(j)
        # print('gt_batch-----------------')
        # print(k)
        # print('gt_batch2-----------------')
        # print(l)
        # print('gt_batch3-----------------')
        # print(np.sum(k, axis=3))


def read_ckpt():
    from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
    path_50 = '/home/gzh/Workspace/Weight/resnet50/resnet_v1_50.ckpt'
    path_101 = '/home/gzh/Workspace/Weight/resnet101/deeplab_resnet_init.ckpt'
    # path = '/home/gzh/Workspace/Weight/resnet101/deeplab_resnet_init.ckpt'
    # print_tensors_in_checkpoint_file(path,None,True)
    from tensorflow.python import pywrap_tensorflow
    reader = pywrap_tensorflow.NewCheckpointReader(path_50)
    var_to_shape_map = reader.get_variable_to_shape_map()
    flag = 0
    for key in sorted(var_to_shape_map):
        print("tensor_name: ", key)
        print("value:", reader.get_tensor(key))
        flag += 1
        if flag > 5: break


def res50_convert():
    ori_path = '/home/gzh/Workspace/Weight/resnet50/resnet_v1_50.ckpt'
    new_path = '/home/gzh/Workspace/Weight/resnet50/deeplab_res50_init.ckpt'
    from tensorflow.python import pywrap_tensorflow
    reader = pywrap_tensorflow.NewCheckpointReader(ori_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    writer = pywrap_tensorflow
    a = tf.train.Saver()
    a.restore()
    for key in var_to_shape_map:
        pass


def deconv(self, input, kernel, output_shape, strides, output_channel, name, reuse=None,
           relu=False,
           padding='SAME',
           biased=False):
    # Verify that the padding is acceptable
    self.validate_padding(padding)
    # Get the number of channels in the input
    input_channel = input.get_shape().as_list()[-1]

    deconvolve = lambda i, k, os: tf.nn.conv2d_transpose(i, k, os, [1, strides[0], strides[1], 1],
                                                         padding=padding)
    with tf.variable_scope(name, reuse=reuse) as scope:

        output_shape = [output_shape[0], output_shape[1], output_shape[2], output_channel]
        f_shape = [kernel[0], kernel[1], output_channel, input_channel]
        filter = self.get_deconv_filter(f_shape, 'weights')
        output = deconvolve(input, filter, output_shape)
        if biased:
            biases = self.make_var('biases', [output_channel])
            output = tf.nn.bias_add(output, biases)
        if relu:
            # ReLU non-linearity
            output = tf.nn.relu(output, name=scope.name)
    return output


def make_deconv_filter(self, name, filter_shape):
    from math import ceil
    width, heigh = filter_shape[0], filter_shape[1]
    f = ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([filter_shape[0], filter_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(filter_shape)
    for i in range(filter_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights, dtype=tf.float32)
    return tf.get_variable(name=name, initializer=init, shape=weights.shape)


if __name__ == '__main__':
    # check2()
    read_ckpt()
