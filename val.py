# _*_ coding:utf-8
import os
import time

import tensorflow as tf
import numpy as np
from models.generator import choose_generator
from utils.data_handle import save_weight, load_weight
from utils.image_process import prepare_label, inv_preprocess, decode_labels
from utils.image_reader import read_labeled_image_list


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
    image_list, label_list, png_list = read_labeled_image_list(args.data_dir, is_val=True,
                                                               valid_image_store_path=args.valid_image_store_path)
    num_val = len(image_list)
    image_name = tf.placeholder(dtype=tf.string)
    label_name = tf.placeholder(dtype=tf.string)
    png_name = tf.placeholder(dtype=tf.string)
    image_batch, label_batch = get_validate_data(image_name, label_name, img_mean)

    print("data load completed!")

    ## load model
    g_net = choose_generator(args.g_name, image_batch)
    raw_output = g_net.terminals[-1]
    predict_batch = g_net.topredict(raw_output, tf.shape(label_batch)[1:3])
    predict_img = tf.write_file(png_name,
                                tf.image.encode_png(tf.cast(tf.squeeze(predict_batch, axis=0), dtype=tf.uint8)))

    labels, logits = convert_to_calculateloss(raw_output, label_batch, args.num_classes)
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
    sum_op = tf.summary.merge_all()
    sum_writer = tf.summary.FileWriter(args.log_dir, max_queue=5)

    sess = tf.Session()
    global_init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()
    sess.run(global_init)
    sess.run(local_init)

    saver = tf.train.Saver(var_list=tf.global_variables())
    _ = load_weight(args.restore_from, saver, sess)

    if not os.path.exists(args.valid_image_store_path):
        os.makedirs(args.valid_image_store_path)

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
