# coding: utf-8
import os

import numpy as np
import tensorflow as tf

from models.generator import choose_generator
from models.discriminator_multitask import choose_discriminator
from utils.data_handle import save_weight, load_weight
from utils.image_process import prepare_label, inv_preprocess, decode_labels
from utils.image_reader import ImageReader


def convert_to_scaling(score_map, num_classes, label_batch, tau=0.9):
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
    gt_batch = tf.clip_by_value(gt_batch, 0. + 1e-8, 1. - 1e-8)
    c = tf.expand_dims(tf.reduce_mean(gt_batch, axis=3), axis=3)
    nor_sum = tf.concat([c for i in range(num_classes)], axis=3)
    gt_batch = gt_batch / nor_sum

    return gt_batch


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
    eps = 1e-8
    print("g_name:", args.g_name)
    print("d_name:", args.d_name)
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
    score_map = g_net.get_output()
    fk_batch = tf.nn.softmax(score_map, dim=-1)
    pre_batch = tf.expand_dims(tf.cast(tf.argmax(fk_batch, axis=-1), tf.uint8), axis=-1)
    gt_batch = tf.image.resize_nearest_neighbor(label_batch, tf.shape(score_map)[1:3])
    gt_batch = tf.where(tf.equal(gt_batch, args.ignore_label), pre_batch, gt_batch)
    gt_batch = convert_to_scaling(fk_batch, args.num_classes, gt_batch)
    x_batch = g_net.get_appointed_layer('generator/image_conv5_3')
    d_fk_net, d_gt_net = choose_discriminator(args.d_name, fk_batch, gt_batch, x_batch)
    d_fk_pred = d_fk_net.get_output()  # fake segmentation result in d
    d_gt_pred = d_gt_net.get_output()  # ground-truth result in d

    label, logits = convert_to_calculateloss(score_map, args.num_classes, label_batch)
    predict_label = tf.argmax(logits, axis=1)
    predict_batch = g_net.topredict(score_map, tf.shape(image_batch)[1:3])
    print("The model has been created!")

    ## get all kinds of variables list
    g_restore_var = [v for v in tf.global_variables() if 'discriminator' not in v.name]
    vgg_restore_var = [v for v in tf.global_variables() if 'discriminator' in v.name and 'image' in v.name]
    g_var = [v for v in tf.trainable_variables() if 'discriminator' not in v.name]
    d_var = [v for v in tf.trainable_variables() if 'discriminator' in v.name and 'image' not in v.name]
    # g_trainable_var = [v for v in g_var if 'beta' not in v.name or 'gamma' not in v.name] #batch_norm training open
    g_trainable_var = g_var
    d_trainable_var = d_var

    ## set loss
    mce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits))
    # g_bce_loss = tf.reduce_mean(tf.log(d_fk_pred + eps))
    g_bce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fk_pred), logits=d_fk_pred)
    g_loss = mce_loss - args.lambd * g_bce_loss
    # d_loss = tf.reduce_mean(tf.constant(-1.0) * [tf.log(d_gt_pred + eps) + tf.log(1. - d_fk_pred + eps)])
    d_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_gt_pred), logits=d_gt_pred) \
             + tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fk_pred), logits=d_fk_pred)

    fk_score_var = tf.reduce_mean(d_fk_pred)
    gt_score_var = tf.reduce_mean(d_gt_pred)
    mce_loss_var, mce_loss_op = tf.metrics.mean(mce_loss)
    g_bce_loss_var, g_bce_loss_op = tf.metrics.mean(g_bce_loss)
    g_loss_var, g_loss_op = tf.metrics.mean(g_loss)
    d_loss_var, d_loss_op = tf.metrics.mean(d_loss)
    iou_var, iou_op = tf.metrics.mean_iou(label, predict_label, args.num_classes)
    accuracy_var, acc_op = tf.metrics.accuracy(label, predict_label)
    metrics_op = tf.group(mce_loss_op, g_bce_loss_op, g_loss_op, d_loss_op, iou_op, acc_op)

    ## set optimizer
    iterstep = tf.placeholder(dtype=tf.float32, shape=[], name='iteration_step')

    base_lr = tf.constant(args.learning_rate, dtype=tf.float32, shape=[])
    lr = tf.scalar_mul(base_lr,
                       tf.pow((1 - iterstep / args.num_steps), args.power))  # learning rate reduce with the time

    g_gradients = tf.train.MomentumOptimizer(learning_rate=lr, momentum=args.momentum).compute_gradients(g_loss,
                                                                                                         g_trainable_var)
    d_gradients = tf.train.MomentumOptimizer(learning_rate=lr * 100, momentum=args.momentum).compute_gradients(d_loss,
                                                                                                               d_trainable_var)
    grad_fk_oi = tf.gradients(d_fk_pred, fk_batch, name='grad_fk_oi')[0]
    grad_gt_oi = tf.gradients(d_gt_pred, gt_batch, name='grad_gt_oi')[0]
    grad_fk_img_oi = tf.gradients(d_fk_pred, image_batch, name='grad_fk_img_oi')[0]
    grad_gt_img_oi = tf.gradients(d_gt_pred, image_batch, name='grad_gt_img_oi')[0]

    train_g_op = tf.train.MomentumOptimizer(learning_rate=lr,
                                            momentum=args.momentum).minimize(g_loss,
                                                                             var_list=g_trainable_var)
    train_d_op = tf.train.MomentumOptimizer(learning_rate=lr * 10,
                                            momentum=args.momentum).minimize(d_loss,
                                                                             var_list=d_trainable_var)

    ## set summary
    vs_image = tf.py_func(inv_preprocess, [image_batch, args.save_num_images, img_mean], tf.uint8)
    vs_label = tf.py_func(decode_labels, [label_batch, args.save_num_images, args.num_classes], tf.uint8)
    vs_predict = tf.py_func(decode_labels, [predict_batch, args.save_num_images, args.num_classes], tf.uint8)
    tf.summary.image(name='image collection_train', tensor=tf.concat(axis=2, values=[vs_image, vs_label, vs_predict]),
                     max_outputs=args.save_num_images)
    tf.summary.scalar('fk_score', tf.reduce_mean(d_fk_pred))
    tf.summary.scalar('gt_score', tf.reduce_mean(d_gt_pred))
    tf.summary.scalar('g_loss_train', g_loss_var)
    tf.summary.scalar('d_loss_train', d_loss_var)
    tf.summary.scalar('mce_loss_train', mce_loss_var)
    tf.summary.scalar('g_bce_loss_train', -1. * g_bce_loss_var)
    tf.summary.scalar('iou_train', iou_var)
    tf.summary.scalar('accuracy_train', accuracy_var)
    tf.summary.scalar('grad_fk_oi', tf.reduce_mean(tf.abs(grad_fk_oi)))
    tf.summary.scalar('grad_gt_oi', tf.reduce_mean(tf.abs(grad_gt_oi)))
    tf.summary.scalar('grad_fk_img_oi', tf.reduce_mean(tf.abs(grad_fk_img_oi)))
    tf.summary.scalar('grad_gt_img_oi', tf.reduce_mean(tf.abs(grad_gt_img_oi)))

    for grad, var in g_gradients + d_gradients:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(args.log_dir, graph=tf.get_default_graph(), max_queue=3)

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
    saver_all = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=2)
    trained_step = 0
    if os.path.exists(args.restore_from + 'checkpoint'):
        trained_step = load_weight(args.restore_from, saver_all, sess)
    else:
        saver_g = tf.train.Saver(var_list=g_restore_var, max_to_keep=2)
        load_weight(args.baseweight_from['g'], saver_g, sess)  # the weight is the completely g model

    threads = tf.train.start_queue_runners(sess, coord)
    print("all setting has been done,training start!")

    ## start training
    def auto_setting_train_steps(mode):
        if mode == 0:
            return 5, 1
        elif mode == 1:
            return 1, 5
        else:
            return 1, 1

    d_train_steps = 5
    g_train_steps = 1
    flags = [0 for i in range(3)]

    for step in range(args.num_steps):
        now_step = int(trained_step) + step if trained_step is not None else step
        feed_dict = {iterstep: step}
        for i in range(d_train_steps):
            _, _ = sess.run([train_d_op, metrics_op], feed_dict)

        for i in range(g_train_steps):
            g_loss_, mce_loss_, g_bce_loss_, d_loss_, _, _ = sess.run(
                [g_loss_var, mce_loss_var, g_bce_loss_var, d_loss_var, train_g_op, metrics_op],
                feed_dict)

        ########################
        fk_score_, gt_score_ = sess.run([fk_score_var, gt_score_var], feed_dict)
        if fk_score_ > 0.48 and fk_score_ < 0.52:
            flags[0] += 1
            flags[1] = flags[2] = 0
        elif gt_score_ - fk_score_ > 0.3:
            flags[1] += 1
            flags[0] = flags[2] = 0
        else:
            flags[2] += 1
            flags[0] = flags[1] = 0
        if max(flags) > 100:
            d_train_steps, g_train_steps = auto_setting_train_steps(flags.index(max(flags)))
        ########################

        if step > 0 and step % args.save_pred_every == 0:
            save_weight(args.restore_from, saver_all, sess, now_step)

        if step % 50 == 0 or step == args.num_steps - 1:
            print('step={} d_loss={} g_loss={} mce_loss={} g_bce_loss_={}'.format(now_step, d_loss_,
                                                                                  g_loss_,
                                                                                  mce_loss_,
                                                                                  g_bce_loss_))
            summary_str = sess.run(summary_op, feed_dict)
            summary_writer.add_summary(summary_str, now_step)
            sess.run(local_init)

    ## end training
    coord.request_stop()
    coord.join(threads)
    print('end....')
