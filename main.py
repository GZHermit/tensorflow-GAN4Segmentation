# coding: utf-8
import argparse
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
time.sleep(2)

import val
import val_include_d
import train_g
import train_d_vgg


def start(args):

    if args.is_val:
        print("Go into the validation stage")
        val_include_d.val(args)
    else:
        print("Go into the train stage")
        train_g.train(args)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """

    BATCH_SIZE = 4
    DATA_DIRECTORY = ['/media/fanyang/workspace/DataSet/voc+sbd/VOC2012/', ]
    IGNORE_LABEL = 255
    IMG_SIZE = (513, 513)
    LEARNING_RATE = 3e-5
    MOMENTUM = 0.9
    NUM_CLASSES = 21
    NUM_STEPS = 50000 + 1
    POWER = 0.9
    RANDOM_SEED = 1234
    IS_VAL = False
    SAVE_NUM_IMAGES = 1
    SAVE_PRED_EVERY = 500
    WEIGHT_DECAY = 0.0005
    D_NAME = 'disc_add_vgg'
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
    parser.add_argument("--save_num_images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save_pred_every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--is_val", type=bool, default=IS_VAL,
                        help="Use the Val")
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

    RESTORE_FROM = './weights/%s/%f/' % (args.d_name, args.learning_rate)
    LOG_DIR = './tblogs/val/%s/%f/' % (args.d_name, args.learning_rate) if args.is_val else './tblogs/train/%s/%f/' % (
        args.d_name, args.learning_rate)
    BASEWEIGHT_FROM = './weights/init/vgg16.npy'
    parser.add_argument("--log_dir", type=str, default=LOG_DIR,
                        help="Where to save tensorboard log of the model.")
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--baseweight_from", type=str, default=BASEWEIGHT_FROM,
                        help="Where base model weight from")
    args = parser.parse_args()
    start(args)
