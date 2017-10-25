# coding: utf-8
import argparse
import os
import time
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# time.sleep(2)

import val
import val_include_d
import train_g_vgg
import train_g_resnet
import train_d_vgg
import train_multitask_vgg


def start(args):
    if args.is_multitask:
        train_multitask_vgg.train(args)
    else:
        if args.is_val:
            print("Go into the validation stage")
            # val_include_d.val(args)
            val.val(args)
        else:
            print("Go into the train stage")
            if args.d_name != 'null':
                train_d_vgg.train(args)
            else:
                if '50' in args.g_name:
                    train_g_resnet.train(args)
                else:
                    train_g_vgg.train(args)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """

    BATCH_SIZE = 1
    DATA_DIRECTORY = ['/home/shared4TB/GZhao/Dataset/VOCdevkit/VOC2012/', ]
    IGNORE_LABEL = 255
    IMG_SIZE = None  # None means we won't use any scaleing or mirroring or resizeing,the input image is origin image.
    LEARNING_RATE = 1e-4
    MOMENTUM = 0.9
    NUM_CLASSES = 21
    NUM_STEPS = 100000 + 1
    POWER = 0.9
    RANDOM_SEED = random.randint(0, 2 ** 31 - 1)
    IS_VAL = False
    IS_MULTITASK = True
    SAVE_NUM_IMAGES = 1
    SAVE_PRED_EVERY = 500
    WEIGHT_DECAY = 0.0003
    D_NAME = 'disc_add_vgg'  # options:disc_add_res50
    G_NAME = 'vgg_32'  # options:vgg_32,vgg_16,vgg_8,res_50
    LAMBD = 0.1

    parser = argparse.ArgumentParser(description="VGG for Semantic Segmentation")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data_dir", type=list, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--ignore_label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--img_size", type=tuple, default=IMG_SIZE,
                        help="Comma_separated string with height and width of images.")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--lambd", type=float, default=LAMBD,
                        help="a constant for constrainting the D-model loss")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random_seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--d_name", type=str, default=D_NAME,
                        help="which d_model can be choosed")
    parser.add_argument("--g_name", type=str, default=G_NAME,
                        help="which g_model can be choosed")
    parser.add_argument("--save_num_images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save_pred_every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--is_val", type=bool, default=IS_VAL,
                        help="Use the Val")
    parser.add_argument("--is_multitask", type=bool, default=IS_MULTITASK,
                        help="train with using the multitask")
    parser.add_argument("--is_training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--not_restore_last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--random_mirror", type=bool, default=True,
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random_scale", type=bool, default=True,
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random_crop", type=bool, default=True,
                        help="Whether to randomly scale the inputs during the training.")

    return parser, parser.parse_args()


if __name__ == '__main__':
    parser, args = get_arguments()
    if args.is_multitask:
        RESTORE_FROM = './weights/is_multi/%s/%s/%f/' % (args.g_name, args.d_name, args.learning_rate)
        LOG_DIR = './tblogs/val/is_multi/%s/%s/%f/' % (
            args.g_name, args.d_name, args.learning_rate) if args.is_val else './tblogs/train/is_multi/%s/%s/%f/' % (
            args.g_name, args.d_name, args.learning_rate)
        VALID_IMAGE_STORE_PATH = './valid_imgs/is_multi/%s/%s/%f/' % (args.g_name, args.d_name, args.learning_rate)
    else:
        RESTORE_FROM = './weights/no_multi/%s/%s/%f/' % (args.g_name, args.d_name, args.learning_rate)
        LOG_DIR = './tblogs/val/no_multi/%s/%s/%f/' % (
            args.g_name, args.d_name, args.learning_rate) if args.is_val else './tblogs/train/no_multi/%s/%s/%f/' % (
            args.g_name, args.d_name, args.learning_rate)
        VALID_IMAGE_STORE_PATH = './valid_imgs/no_multi/%s/%s/%f/' % (args.g_name, args.d_name, args.learning_rate)
    BASEWEIGHT_FROM = {'res50': '/home/shared4TB/GZhao/Weights/resnet_v1_50.ckpt',
                       'vgg16': '/home/shared4TB/GZhao/Weights/vgg16.npy',
                       'g': '/home/daixl/GZHermit/weights/vgg_16/disc_add_vgg/0.000100'}
    parser.add_argument("--log_dir", type=str, default=LOG_DIR,
                        help="Where to save tensorboard log of the model.")
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--valid_image_store_path", type=str, default=VALID_IMAGE_STORE_PATH,
                        help="Where store valid image files")
    parser.add_argument("--baseweight_from", type=dict, default=BASEWEIGHT_FROM,
                        help="Where base model weight from")
    args = parser.parse_args()
    start(args)
