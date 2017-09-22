# _*_ coding:utf-8
import time

import tensorflow as tf
import numpy as np
from models.generator import Generator
from models.discriminator import Discriminator_add_vgg
from utils.data_handle import save_weight, load_weight
from utils.image_process import prepare_label, inv_preprocess, decode_labels
from utils.image_reader import read_labeled_image_list


def convert_to_scaling(score_map, num_classes, label_batch, tau=0.9):
    score_map_max = tf.reduce_max(score_map, axis=3, keep_dims=False)
    y_il = tf.maximum(score_map_max, tf.constant(tau, tf.float32, label_batch.get_shape().as_list()[:-1]))
    _s_il = 1.0 - score_map_max
    _y_il = 1.0 - y_il
    a = tf.expand_dims(tf.div(_y_il, _s_il), axis=3)
    y_ic = tf.concat([a for i in range(num_classes)], axis=3)
    y_ic = tf.multiply(score_map, y_ic)
    b = tf.expand_dims(y_il, axis=3)
    y_il_ = tf.concat([b for i in range(num_classes)], axis=3)
    lab_hot = tf.squeeze(tf.one_hot(label_batch, num_classes, dtype=tf.float32), axis=3)
    gt_batch = tf.where(tf.equal(lab_hot, 1.), y_il_, y_ic)
    gt_batch = tf.clip_by_value(gt_batch, 0., 1.)

    return gt_batch

def convert_to_calculateloss(raw_output, label_batch, num_classes):
    raw_output = tf.image.resize_bilinear(raw_output, tf.shape(label_batch)[1:3])
    raw_groundtruth = tf.reshape(tf.squeeze(label_batch, squeeze_dims=[3]), [-1, ])
    raw_prediction = tf.reshape(raw_output, [-1, num_classes])

    indices = tf.squeeze(tf.where(tf.less_equal(raw_groundtruth, num_classes - 1)), 1)
    label = tf.cast(tf.gather(raw_groundtruth, indices), tf.int32)  # [?, ]
    logits = tf.gather(raw_prediction, indices)  # [?, num_classes]

    return label, logits


def get_validate_data(image_name, label_name, img_mean):
    img = tf.read_file(image_name)
    img = tf.image.decode_jpeg(img, channels=3)
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    img -= img_mean
    img = tf.expand_dims(img, axis=0)

    label = tf.read_file(label_name)
    label = tf.image.decode_png(label, channels=1)
    label = tf.expand_dims(label, axis=0)

    return img, label


def val(args):
    ## set hyparameter
    img_mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    tf.set_random_seed(args.random_seed)

    ## load data
    image_list, label_list, png_list = read_labeled_image_list(args.data_dir, is_val=True)
    num_val = len(image_list)
    image_name = tf.placeholder(dtype=tf.string)
    label_name = tf.placeholder(dtype=tf.string)
    png_name = tf.placeholder(dtype=tf.string)
    image_batch, label_batch = get_validate_data(image_name, label_name, img_mean)

    print("data load completed!")

    ## load model
    g_net = Generator(inputs={'data': image_batch})
    score_map = g_net.get_output()
    fk_batch = tf.nn.softmax(score_map, dim=-1)
    gt_batch = tf.image.resize_nearest_neighbor(label_batch, score_map.get_shape()[1:3])
    gt_batch = convert_to_scaling(fk_batch, args.num_classes, gt_batch)
    x_batch = tf.train.batch([(image_batch + img_mean) / 255., ], args.batch_size)  # normalization
    d_fk_net = Discriminator_add_vgg({'seg': fk_batch, 'data': x_batch})
    d_gt_net = Discriminator_add_vgg({'seg': gt_batch, 'data': x_batch}, reuse=True)
    d_fk_pred = d_fk_net.get_output()  # fake segmentation result in d
    d_gt_pred = d_gt_net.get_output()  # ground-truth result in d

    predict_batch = g_net.topredict(score_map, tf.shape(label_batch)[1:3])
    predict_img = tf.write_file(png_name,
                                tf.image.encode_png(tf.cast(tf.squeeze(predict_batch, axis=0), dtype=tf.uint8)))

    labels, logits = convert_to_calculateloss(score_map, label_batch, args.num_classes)
    pre_labels = tf.argmax(logits, 1)

    print("Model load completed!")

    iou, iou_op = tf.metrics.mean_iou(labels, pre_labels, args.num_classes, name='iou')
    acc, acc_op = tf.metrics.accuracy(labels, pre_labels)
    m_op = tf.group(iou_op, acc_op)

    image = tf.py_func(inv_preprocess, [image_batch, args.save_num_images, img_mean], tf.uint8)
    label = tf.py_func(decode_labels, [label_batch, ], tf.uint8)
    pred = tf.py_func(decode_labels, [predict_batch, ], tf.uint8)
    tf.summary.image(name='img_collection_val', tensor=tf.concat([image, label, pred], 2))
    tf.summary.scalar(name='iou_val', tensor=iou)
    tf.summary.scalar(name='acc_val', tensor=acc)
    tf.summary.scalar('fk_score', tf.reduce_mean(d_fk_pred))
    tf.summary.scalar('gt_score', tf.reduce_mean(d_gt_pred))
    sum_op = tf.summary.merge_all()
    sum_writer = tf.summary.FileWriter(args.log_dir, max_queue=20)

    sess = tf.Session()
    global_init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()
    sess.run(global_init)
    sess.run(local_init)

    saver = tf.train.Saver(var_list=tf.global_variables())
    _ = load_weight(args.restore_from, saver, sess)

    print("validation begining")

    for step in range(num_val):
        it = time.time()
        dict = {image_name: image_list[step], label_name: label_list[step], png_name: png_list[step]}
        _, _, = sess.run([m_op, predict_img], dict)
        if step % 50 == 0 or step == num_val - 1:
            summ = sess.run(sum_op, dict)
            sum_writer.add_summary(summ, step)
            print("step:{},time:{}".format(step, time.time() - it))

    print("end......")
