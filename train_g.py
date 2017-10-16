# coding: utf-8
import os

import numpy as np
import tensorflow as tf

from models.generator import choose_generator
from utils.data_handle import save_weight, load_weight
from utils.image_process import prepare_label, inv_preprocess, decode_labels
from utils.image_reader import ImageReader


def convert_to_calculateloss(score_map, num_classes, label_batch):
    label_proc = prepare_label(label_batch, tf.shape(score_map)[1:3],
                               num_classes=num_classes, one_hot=False)  # [batch_size, h, w]
    raw_groundtruth = tf.reshape(label_proc, [-1, ])
    raw_prediction = tf.reshape(score_map, [-1, num_classes])
    indices = tf.squeeze(tf.where(tf.less_equal(raw_groundtruth, num_classes - 1)), 1)
    label = tf.cast(tf.gather(raw_groundtruth, indices), tf.int32)  # [?, ]
    logits = tf.gather(raw_prediction, indices)  # [?, num_classes]

    return label, logits


def train(args):
    ## set hyparameter
    img_mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    tf.set_random_seed(args.random_seed)
    coord = tf.train.Coordinator()
    print("g_model_name:", args.g_name)
    print("lambda:", args.lambd)
    print("learning_rate:", args.learning_rate)
    print("is_val:", args.is_val)
    print("---------------------------------")

    ## load data
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.img_size,
            args.random_scale,
            args.random_mirror,
            args.random_crop,
            args.ignore_label,
            args.is_val,
            img_mean,
            coord)
        image_batch, label_batch = reader.dequeue(args.batch_size)
        print("Data is ready!")

    ## load model
    g_net = choose_generator(args.g_name, image_batch)
    score_map = g_net.get_output()  # [batch_size, h, w, num_classes]

    label, logits = convert_to_calculateloss(score_map, args.num_classes, label_batch)
    predict_label = tf.argmax(logits, axis=1)
    predict_batch = g_net.topredict(score_map, tf.shape(image_batch)[1:3])
    print("The model has been created!")

    ## get all kinds of variables list
    g_restore_var = [v for v in tf.global_variables() if 'generator' in v.name and 'image' in v.name]
    g_trainable_var = tf.trainable_variables()

    ## set loss
    g_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits))
    g_loss_var, g_loss_op = tf.metrics.mean(g_loss)
    iou_var, iou_op = tf.metrics.mean_iou(label, predict_label, args.num_classes)
    accuracy_var, acc_op = tf.metrics.accuracy(label, predict_label)
    metrics_op = tf.group(g_loss_op, iou_op, acc_op)

    ## set optimizer
    iterstep = tf.placeholder(dtype=tf.float32, shape=[], name='iteration_step')

    base_lr = tf.constant(args.learning_rate, dtype=tf.float32, shape=[])
    lr = tf.scalar_mul(base_lr,
                       tf.pow((1 - iterstep / args.num_steps), args.power))  # learning rate reduce with the time

    # g_gradients = tf.train.MomentumOptimizer(learning_rate=lr, momentum=args.momentum).compute_gradients(g_loss,
    #                                                                                                      g_trainable_var)
    train_g_op = tf.train.MomentumOptimizer(learning_rate=lr,
                                            momentum=args.momentum).minimize(g_loss,
                                                                             var_list=g_trainable_var)
    train_all_op = train_g_op

    ## set summary
    vs_image = tf.py_func(inv_preprocess, [image_batch, args.save_num_images, img_mean], tf.uint8)
    vs_label = tf.py_func(decode_labels, [label_batch, args.save_num_images, args.num_classes], tf.uint8)
    vs_predict = tf.py_func(decode_labels, [predict_batch, args.save_num_images, args.num_classes], tf.uint8)
    tf.summary.image(name='image collection_train', tensor=tf.concat(axis=2, values=[vs_image, vs_label, vs_predict]),
                     max_outputs=args.save_num_images)

    tf.summary.scalar('g_loss_train', g_loss_var)
    tf.summary.scalar('iou_train', iou_var)
    tf.summary.scalar('accuracy_train', accuracy_var)
    # for grad, var in g_gradients:
    #     tf.summary.histogram(var.op.name + "/gradients", grad)
    #
    # for var in tf.trainable_variables():
    #     tf.summary.histogram(var.op.name + "/values", var)

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(args.log_dir, graph=tf.get_default_graph(), max_queue=5)

    ## set session
    print("GPU index:" + str(os.environ['CUDA_VISIBLE_DEVICES']))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    global_init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()
    sess.run(global_init)
    sess.run(local_init)

    ## set saver
    saver_all = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=50)
    trained_step = 0
    if os.path.exists(args.restore_from + 'checkpoint'):
        trained_step = load_weight(args.restore_from, saver_all, sess)
    else:
        load_weight(args.baseweight_from, g_restore_var, sess)

    threads = tf.train.start_queue_runners(sess, coord)
    print("all setting has been done,training start!")

    ## start training
    for step in range(args.num_steps):
        now_step = int(trained_step) + step if trained_step is not None else step
        feed_dict = {iterstep: now_step}
        g_loss_, _, _ = sess.run([g_loss_var, train_all_op, metrics_op], feed_dict)

        if step > 0 and step % args.save_pred_every == 0:
            save_weight(args.restore_from, saver_all, sess, now_step)

        if step % 50 == 0 or step == args.num_steps - 1:
            print('step={} g_loss={}'.format(now_step, g_loss_))
            summary_str = sess.run(summary_op, feed_dict)
            summary_writer.add_summary(summary_str, now_step)
            sess.run(local_init)

    ## end training
    coord.request_stop()
    coord.join(threads)
    print('end....')
