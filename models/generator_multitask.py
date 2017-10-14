# coding:utf-8
from models.network import NetWork
import tensorflow as tf


class Generator_vgg16(NetWork):
    def setup(self, is_training, num_classes):
        name = 'generator/'
        (self.feed('data')
         .conv([3, 3], 64, [1, 1], biased=True, relu=True, name=name + 'conv5_4'))

    def topredict(self, raw_output, origin_shape=None):
        pass


class Generator_resnet50(NetWork):
    def setup(self, is_training, num_classes):
        (self.feed('data')
         .conv([7, 7], 64, [2, 2], biased=False, relu=False, name='conv1')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1')
         .max_pool([3, 3], [2, 2], name='pool1')
         .conv([1, 1], 256, [1, 1], biased=False, relu=False, name='res2a_branch1')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch1'))

        (self.feed('pool1')
         .conv([1, 1], 64, [1, 1], biased=False, relu=False, name='res2a_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2a_branch2a')
         .conv([3, 3], 64, [1, 1], biased=False, relu=False, name='res2a_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2a_branch2b')
         .conv([1, 1], 256, [1, 1], biased=False, relu=False, name='res2a_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch2c'))

        (self.feed('bn2a_branch1',
                   'bn2a_branch2c')
         .add(name='res2a')
         .relu(name='res2a_relu')
         .conv([1, 1], 64, [1, 1], biased=False, relu=False, name='res2b_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2b_branch2a')
         .conv([3, 3], 64, [1, 1], biased=False, relu=False, name='res2b_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2b_branch2b')
         .conv([1, 1], 256, [1, 1], biased=False, relu=False, name='res2b_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn2b_branch2c'))

        (self.feed('res2a_relu',
                   'bn2b_branch2c')
         .add(name='res2b')
         .relu(name='res2b_relu')
         .conv([1, 1], 64, [1, 1], biased=False, relu=False, name='res2c_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2c_branch2a')
         .conv([3, 3], 64, [1, 1], biased=False, relu=False, name='res2c_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2c_branch2b')
         .conv([1, 1], 256, [1, 1], biased=False, relu=False, name='res2c_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn2c_branch2c'))

        (self.feed('res2b_relu',
                   'bn2c_branch2c')
         .add(name='res2c')
         .relu(name='res2c_relu')
         .conv([1, 1], 512, [2, 2], biased=False, relu=False, name='res3a_branch1')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn3a_branch1'))

        (self.feed('res2c_relu')
         .conv([1, 1], 128, [2, 2], biased=False, relu=False, name='res3a_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3a_branch2a')
         .conv([3, 3], 128, [1, 1], biased=False, relu=False, name='res3a_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3a_branch2b')
         .conv([1, 1], 512, [1, 1], biased=False, relu=False, name='res3a_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn3a_branch2c'))

        (self.feed('bn3a_branch1',
                   'bn3a_branch2c')
         .add(name='res3a')
         .relu(name='res3a_relu')
         .conv([1, 1], 128, [1, 1], biased=False, relu=False, name='res3b_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b_branch2a')
         .conv([3, 3], 128, [1, 1], biased=False, relu=False, name='res3b_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b_branch2b')
         .conv([1, 1], 512, [1, 1], biased=False, relu=False, name='res3b_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn3b_branch2c'))

        (self.feed('res3a_relu',
                   'bn3b_branch2c')
         .add(name='res3b')
         .relu(name='res3b_relu')
         .conv([1, 1], 128, [1, 1], biased=False, relu=False, name='res3c_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3c_branch2a')
         .conv([3, 3], 128, [1, 1], biased=False, relu=False, name='res3c_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3c_branch2b')
         .conv([1, 1], 512, [1, 1], biased=False, relu=False, name='res3c_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn3c_branch2c'))

        (self.feed('res3b_relu',
                   'bn3c_branch2c')
         .add(name='res3c')
         .relu(name='res3c_relu')
         .conv([1, 1], 128, [1, 1], biased=False, relu=False, name='res3d_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3d_branch2a')
         .conv([3, 3], 128, [1, 1], biased=False, relu=False, name='res3d_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3d_branch2b')
         .conv([1, 1], 512, [1, 1], biased=False, relu=False, name='res3d_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn3d_branch2c'))

        (self.feed('res3c_relu',
                   'bn3d_branch2c')
         .add(name='res3d')
         .relu(name='res3d_relu')
         .conv([1, 1], 1024, [2, 2], biased=False, relu=False, name='res4a_branch1')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4a_branch1'))

        (self.feed('res3d_relu')
         .conv([1, 1], 256, [2, 2], biased=False, relu=False, name='res4a_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4a_branch2a')
         .conv([3, 3], 256, [1, 1], padding='SAME', biased=False, relu=False, name='res4a_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4a_branch2b')
         .conv([1, 1], 1024, [1, 1], biased=False, relu=False, name='res4a_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4a_branch2c'))

        # when we need to train model with output_stride=8,the rate of conv change from 1 to 2 in block3
        (self.feed('bn4a_branch1',
                   'bn4a_branch2c')
         .add(name='res4a')
         .relu(name='res4a_relu')
         .conv([1, 1], 256, [1, 1], biased=False, relu=False, name='res4b_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b_branch2a')
         .conv([3, 3], 256, [1, 1], padding='SAME', biased=False, relu=False, name='res4b_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b_branch2b')
         .conv([1, 1], 1024, [1, 1], biased=False, relu=False, name='res4b_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b_branch2c'))

        (self.feed('res4a_relu',
                   'bn4b_branch2c')
         .add(name='res4b')
         .relu(name='res4b_relu')
         .conv([1, 1], 256, [1, 1], biased=False, relu=False, name='res4c_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4c_branch2a')
         .conv([3, 3], 256, [1, 1], padding='SAME', biased=False, relu=False, name='res4c_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4c_branch2b')
         .conv([1, 1], 1024, [1, 1], biased=False, relu=False, name='res4c_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4c_branch2c'))

        (self.feed('res4b_relu',
                   'bn4c_branch2c')
         .add(name='res4c')
         .relu(name='res4c_relu')
         .conv([1, 1], 256, [1, 1], biased=False, relu=False, name='res4d_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4d_branch2a')
         .conv([3, 3], 256, [1, 1], padding='SAME', biased=False, relu=False, name='res4d_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4d_branch2b')
         .conv([1, 1], 1024, [1, 1], biased=False, relu=False, name='res4d_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4d_branch2c'))

        (self.feed('res4c_relu',
                   'bn4d_branch2c')
         .add(name='res4d')
         .relu(name='res4d_relu')
         .conv([1, 1], 256, [1, 1], biased=False, relu=False, name='res4e_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4e_branch2a')
         .conv([3, 3], 256, [1, 1], padding='SAME', biased=False, relu=False, name='res4e_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4e_branch2b')
         .conv([1, 1], 1024, [1, 1], biased=False, relu=False, name='res4e_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4e_branch2c'))

        (self.feed('res4d_relu',
                   'bn4e_branch2c')
         .add(name='res4e')
         .relu(name='res4e_relu')
         .conv([1, 1], 256, [1, 1], biased=False, relu=False, name='res4f_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4f_branch2a')
         .conv([3, 3], 256, [1, 1], padding='SAME', biased=False, relu=False, name='res4f_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4f_branch2b')
         .conv([1, 1], 1024, [1, 1], biased=False, relu=False, name='res4f_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4f_branch2c'))

        (self.feed('res4e_relu',
                   'bn4f_branch2c')
         .add(name='res4f')
         .relu(name='res4f_relu')
         .conv([1, 1], 2048, [1, 1], biased=False, relu=False, name='res5a_branch1')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn5a_branch1'))

        (self.feed('res4e_relu')
         .conv([1, 1], 512, [1, 1], biased=False, relu=False, name='res5a_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5a_branch2a')
         .atrous_conv([3, 3], 512, 2, padding='SAME', biased=False, relu=False, name='res5a_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5a_branch2b')
         .conv([1, 1], 2048, [1, 1], biased=False, relu=False, name='res5a_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn5a_branch2c'))

        (self.feed('bn5a_branch1',
                   'bn5a_branch2c')
         .add(name='res5a')
         .relu(name='res5a_relu')
         .conv([1, 1], 512, [1, 1], biased=False, relu=False, name='res5b_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5b_branch2a')
         .conv([3, 3], 512, [1, 1], padding='SAME', biased=False, relu=False, name='res5b_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5b_branch2b')
         .conv([1, 1], 2048, [1, 1], biased=False, relu=False, name='res5b_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn5b_branch2c'))

        (self.feed('res5a_relu',
                   'bn5b_branch2c')
         .add(name='res5b')
         .relu(name='res5b_relu')
         .conv([1, 1], 512, [1, 1], biased=False, relu=False, name='res5c_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5c_branch2a')
         .conv([3, 3], 512, [1, 1], padding='SAME', biased=False, relu=False, name='res5c_branch2b')
         .batch_normalization(activation_fn=tf.nn.relu, name='bn5c_branch2b', is_training=is_training)
         .conv([1, 1], 2048, [1, 1], biased=False, relu=False, name='res5c_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn5c_branch2c'))

        (self.feed('res5b_relu',
                   'bn5c_branch2c')
         .add(name='res5c')
         .relu(name='res5c_relu'))

        (self.feed('fc_res8c_relu')
         .conv([3, 3], 256, 6, padding='SAME', relu=False, name='fc1_voc12_c0')
         .batch_normalization(is_training=is_training, activation_fn=None, name='fc1_voc12_bn0'))

        (self.feed('fc_res8c_relu')
         .conv([3, 3], 256, 12, padding='SAME', relu=False, name='fc1_voc12_c1')
         .batch_normalization(is_training=is_training, activation_fn=None, name='fc1_voc12_bn1'))

        (self.feed('fc_res8c_relu')
         .conv([3, 3], 256, 18, padding='SAME', relu=False, name='fc1_voc12_c2')
         .batch_normalization(is_training=is_training, activation_fn=None, name='fc1_voc12_bn2'))

        (self.feed('fc_res8c_relu')
         .conv([1, 1], 256, [1, 1], relu=False, name='fc1_voc12_c3')
         .batch_normalization(is_training=is_training, activation_fn=None, name='fc1_voc12_bn3'))

        layer = self.get_appointed_layer('fc_res8c_relu')
        new_shape = tf.shape(layer)[1:3]
        (self.feed('fc_res8c_relu')
         .global_average_pooling(name='fc1_voc12_mp0')
         .conv([1, 1], 256, [1, 1], relu=False, name='fc1_voc12_c4')
         .batch_normalization(is_training=is_training, activation_fn=None, name='fc1_voc12_bn4')
         .resize(new_shape, name='fc1_voc12_bu0'))

        (self.feed('fc1_voc12_bn0',
                   'fc1_voc12_bn1',
                   'fc1_voc12_bn2',
                   'fc1_voc12_bn3',
                   'fc1_voc12_bu0')
         .concat(axis=3, name='fc1_voc12'))

        (self.feed('fc1_voc12')
         .conv([1, 1], 256, [1, 1], relu=False, name='fc2_voc12_c0')
         .batch_normalization(is_training=is_training, activation_fn=None, name='fc2_voc12_bn0')
         .conv([1, 1], num_classes, [1, 1], relu=False, name='fc2_voc12_c1'))

    def topredict(self, raw_output, origin_shape=None):
        pass
