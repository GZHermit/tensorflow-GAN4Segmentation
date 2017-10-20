# coding:utf-8
from models.network import NetWork
import tensorflow as tf


def choose_generator(g_name, image_batch):
    if '32' in g_name:
        return Generator_vgg_32({'data': image_batch})
    elif '16' in g_name:
        return Generator_vgg_16({'data': image_batch})
    elif '8' in g_name:
        return Generator_vgg_8({'data': image_batch})
    elif '50' in g_name:
        return Generator_resnet50({'data': image_batch})


class Generator_vgg_32(NetWork):
    def setup(self, is_training, num_classes):
        name = 'generator/'
        (self.feed('data')
         .conv([3, 3], 64, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv1_1')
         .conv([3, 3], 64, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv1_2')
         .max_pool([2, 2], [2, 2], name=name + 'image_pool1')
         .conv([3, 3], 128, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv2_1')
         .conv([3, 3], 128, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv2_2')
         .max_pool([2, 2], [2, 2], name=name + 'image_pool2')
         .conv([3, 3], 256, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv3_1')
         .conv([3, 3], 256, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv3_2')
         .conv([3, 3], 256, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv3_3')
         .max_pool([2, 2], [2, 2], name=name + 'image_pool3')
         .conv([3, 3], 512, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv4_1')
         .conv([3, 3], 512, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv4_2')
         .conv([3, 3], 512, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv4_3')
         .max_pool([2, 2], [2, 2], name=name + 'image_pool4')
         .conv([3, 3], 512, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv5_1')
         .conv([3, 3], 512, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv5_2')
         .conv([3, 3], 512, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv5_3')
         .max_pool([2, 2], [2, 2], name=name + 'image_pool5')
         .conv([7, 7], 4096, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_fc6')
         .dropout(keep_prob=0.5, name=name + 'drop6')
         .conv([1, 1], 4096, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_fc7')
         .dropout(keep_prob=0.5, name=name + 'drop7')
         .conv([1, 1], num_classes, [1, 1], reuse=self.reuse, biased=True, relu=False, name=name + 'score_fr'))
        origin_shape = tf.multiply(tf.shape(self.layers['data']), tf.convert_to_tensor([1, 1, 1, 7]))

        (self.feed(name + 'score_fr')
         .resize(origin_shape[1:3], name=name + 'upscore32'))
        # .deconv([64, 64], origin_shape, [32, 32], num_classes, reuse=self.reuse, biased=False, relu=False,
        #         name=name + 'upscore32'))

    def topredict(self, raw_output, origin_shape=None):
        raw_output = tf.image.resize_bilinear(raw_output, origin_shape)
        raw_output = tf.argmax(raw_output, axis=3)
        prediction = tf.expand_dims(raw_output, dim=3)
        return prediction


class Generator_vgg_16(NetWork):
    def setup(self, is_training, num_classes):
        name = 'generator/'
        (self.feed('data')
         .conv([3, 3], 64, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv1_1')
         .conv([3, 3], 64, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv1_2')
         .max_pool([2, 2], [2, 2], name=name + 'image_pool1')
         .conv([3, 3], 128, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv2_1')
         .conv([3, 3], 128, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv2_2')
         .max_pool([2, 2], [2, 2], name=name + 'image_pool2')
         .conv([3, 3], 256, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv3_1')
         .conv([3, 3], 256, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv3_2')
         .conv([3, 3], 256, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv3_3')
         .max_pool([2, 2], [2, 2], name=name + 'image_pool3')
         .conv([3, 3], 512, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv4_1')
         .conv([3, 3], 512, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv4_2')
         .conv([3, 3], 512, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv4_3')
         .max_pool([2, 2], [2, 2], name=name + 'image_pool4')
         .conv([3, 3], 512, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv5_1')
         .conv([3, 3], 512, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv5_2')
         .conv([3, 3], 512, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv5_3')
         .max_pool([2, 2], [2, 2], name=name + 'image_pool5')
         .conv([7, 7], 4096, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_fc6')
         .dropout(keep_prob=0.5, name=name + 'drop6')
         .conv([1, 1], 4096, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_fc7')
         .dropout(keep_prob=0.5, name=name + 'drop7')
         .conv([1, 1], num_classes, [1, 1], reuse=self.reuse, biased=True, relu=False, name=name + 'score_fr'))

        (self.feed(name + 'image_pool4')
         .conv([1, 1], num_classes, [1, 1], reuse=self.reuse, biased=True, relu=False, name=name + 'score_pool4'))
        pool_shape = tf.shape(self.layers[name + 'score_pool4'])

        (self.feed(name + 'score_fr')
         .resize(pool_shape[1:3], name=name + 'upscore2'))
        # .deconv([4, 4], pool_shape, [2, 2], num_classes, reuse=self.reuse, biased=False, relu=False,
        #         name=name + 'upscore2'))
        (self.feed(name + 'upscore2', name + 'score_pool4')
         .add(name=name + 'fuse_pool4'))
        origin_shape = tf.multiply(tf.shape(self.layers['data']), tf.convert_to_tensor([1, 1, 1, 7]))

        (self.feed(name + 'fuse_pool4')
         .resize(origin_shape[1:3], name=name + 'upscore16'))
        # .deconv([32, 32], origin_shape, [16, 16], num_classes, reuse=self.reuse, biased=False, relu=False,
        #         name=name + 'upscore16'))

    def topredict(self, raw_output, origin_shape=None):
        raw_output = tf.image.resize_bilinear(raw_output, origin_shape)
        raw_output = tf.argmax(raw_output, axis=3)
        prediction = tf.expand_dims(raw_output, dim=3)
        return prediction


class Generator_vgg_8(NetWork):
    def setup(self, is_training, num_classes):
        name = 'generator/'
        (self.feed('data')
         .conv([3, 3], 64, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv1_1')
         .conv([3, 3], 64, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv1_2')
         .max_pool([2, 2], [2, 2], name=name + 'image_pool1')
         .conv([3, 3], 128, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv2_1')
         .conv([3, 3], 128, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv2_2')
         .max_pool([2, 2], [2, 2], name=name + 'image_pool2')
         .conv([3, 3], 256, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv3_1')
         .conv([3, 3], 256, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv3_2')
         .conv([3, 3], 256, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv3_3')
         .max_pool([2, 2], [2, 2], name=name + 'image_pool3')
         .conv([3, 3], 512, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv4_1')
         .conv([3, 3], 512, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv4_2')
         .conv([3, 3], 512, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv4_3')
         .max_pool([2, 2], [2, 2], name=name + 'image_pool4')
         .conv([3, 3], 512, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv5_1')
         .conv([3, 3], 512, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv5_2')
         .conv([3, 3], 512, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv5_3')
         .max_pool([2, 2], [2, 2], name=name + 'image_pool5')
         .conv([7, 7], 4096, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_fc6')
         .dropout(keep_prob=0.5, name=name + 'drop6')
         .conv([1, 1], 4096, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_fc7')
         .dropout(keep_prob=0.5, name=name + 'drop7')
         .conv([1, 1], num_classes, [1, 1], reuse=self.reuse, biased=True, relu=False, name=name + 'score_fr'))

        (self.feed(name + 'image_pool3')
         .conv([1, 1], num_classes, [1, 1], reuse=self.reuse, biased=True, relu=False, name=name + 'score_pool3'))
        pool3_shape = tf.shape(self.layers[name + 'score_pool3'])

        (self.feed(name + 'image_pool4')
         .conv([1, 1], num_classes, [1, 1], reuse=self.reuse, biased=True, relu=False, name=name + 'score_pool4'))
        pool4_shape = tf.shape(self.layers[name + 'score_pool4'])

        (self.feed(name + 'score_fr')
         .resize(pool4_shape[1:3], name=name + 'upscore2'))
        # .deconv([4, 4], pool4_shape, [2, 2], num_classes, reuse=self.reuse, biased=False, relu=False,
        #         name=name + 'upscore2'))

        (self.feed(name + 'upscore2', name + 'score_pool4')
         .add(name=name + 'fuse_pool4')
         .resize(pool3_shape[1:3], name=name + 'upscore_pool4'))
        # .deconv([4, 4], pool3_shape, [2, 2], num_classes, reuse=self.reuse, biased=False, relu=False,
        #         name=name + 'upscore_pool4'))

        origin_shape = tf.multiply(tf.shape(self.layers['data']), tf.convert_to_tensor([1, 1, 1, 7]))
        (self.feed(name + 'upscore_pool4', name + 'score_pool3')
         .add(name=name + 'fuse_pool3')
         .resize(origin_shape[1:3], name=name + 'upscore8'))
        # .deconv([16, 16], origin_shape, [8, 8], num_classes, reuse=self.reuse, biased=False, relu=False,
        #         name=name + 'upscore8'))

    def topredict(self, raw_output, origin_shape=None):
        raw_output = tf.image.resize_bilinear(raw_output, origin_shape)
        raw_output = tf.argmax(raw_output, axis=3)
        prediction = tf.expand_dims(raw_output, dim=3)
        return prediction


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
         .conv([1, 1], 2048, [2, 2], biased=False, relu=False, name='res5a_branch1')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn5a_branch1'))

        (self.feed('res4e_relu')
         .conv([1, 1], 512, [2, 2], biased=False, relu=False, name='res5a_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5a_branch2a')
         .conv([3, 3], 512, [1, 1], padding='SAME', biased=False, relu=False, name='res5a_branch2b')
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
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5c_branch2b')
         .conv([1, 1], 2048, [1, 1], biased=False, relu=False, name='res5c_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn5c_branch2c'))

        (self.feed('res5b_relu',
                   'bn5c_branch2c')
         .add(name='res5c')
         .relu(name='res5c_relu')
         .atrous_conv([3, 3], num_classes, 6, padding='SAME', relu=False, name='fc1_voc12_c0'))

        (self.feed('res5c_relu')
         .atrous_conv([3, 3], num_classes, 12, padding='SAME', relu=False, name='fc1_voc12_c1'))

        (self.feed('res5c_relu')
         .atrous_conv([3, 3], num_classes, 18, padding='SAME', relu=False, name='fc1_voc12_c2'))

        (self.feed('res5c_relu')
         .atrous_conv([3, 3], num_classes, 24, padding='SAME', relu=False, name='fc1_voc12_c3'))

        (self.feed('fc1_voc12_c0',
                   'fc1_voc12_c1',
                   'fc1_voc12_c2',
                   'fc1_voc12_c3')
         .add(name='fc1_voc12'))

    def topredict(self, raw_output, origin_shape=None):
        raw_output = tf.image.resize_bilinear(raw_output, origin_shape)
        raw_output = tf.argmax(raw_output, axis=3)
        prediction = tf.expand_dims(raw_output, dim=3)
        return prediction
