# coding:utf-8
from models.network import NetWork
import tensorflow as tf


def choose_generator(g_name, image_batch):
    if '32' in g_name:
        return Generator_32({'data': image_batch})
    elif '16' in g_name:
        return Generator_16({'data': image_batch})
    elif '8' in g_name:
        return Generator_8({'data': image_batch})


class Generator_32(NetWork):
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
        print('data:', self.layers['data'].get_shape().as_list())
        origin_shape = tf.multiply(tf.shape(self.layers['data']), tf.convert_to_tensor([1, 1, 1, 7]))

        (self.feed(name + 'score_fr')
         .deconv([64, 64], origin_shape, [32, 32], num_classes, reuse=self.reuse, biased=False, relu=False,
                 name=name + 'upscore32'))

    def topredict(self, raw_output, origin_shape=None):
        # raw_output = tf.image.resize_bilinear(raw_output, origin_shape)
        raw_output = tf.argmax(raw_output, axis=3)
        prediction = tf.expand_dims(raw_output, dim=3)
        return prediction


class Generator_16(NetWork):
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
        pool_shape = tf.shape(self.layers[name + 'image_pool4'])
        (self.feed(name + 'score_fr')
         .deconv([4, 4], pool_shape, [2, 2], num_classes, reuse=self.reuse, biased=True, relu=False,
                 name=name + 'upscore2'))
        (self.feed(name + 'score_fr', name + 'score_pool4')
         .add(name=name + 'fuse_pool4'))
        origin_shape = tf.multiply(tf.shape(self.layers['data']), tf.convert_to_tensor([1, 1, 1, 7]))
        (self.feed(name + 'fuse_pool4')
         .deconv([32, 32], origin_shape, [16, 16], num_classes, reuse=self.reuse, biased=True, relu=False,
                 name=name + 'upscore16'))

    def topredict(self, raw_output, origin_shape=None):
        # raw_output = tf.image.resize_bilinear(raw_output, origin_shape)
        raw_output = tf.argmax(raw_output, axis=3)
        prediction = tf.expand_dims(raw_output, dim=3)
        return prediction


class Generator_8(NetWork):
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
         .conv([1, 1], num_classes, [1, 1], reuse=self.reuse, biased=False, relu=False, name=name + 'score_pool3'))
        pool3_shape = tf.shape(self.layers[name + 'image_pool3'])

        (self.feed(name + 'image_pool4')
         .conv([1, 1], num_classes, [1, 1], reuse=self.reuse, biased=False, relu=False, name=name + 'score_pool4'))
        pool4_shape = tf.shape(self.layers[name + 'image_pool4'])

        (self.feed(name + 'score_fr')
         .deconv([4, 4], pool4_shape, [2, 2], num_classes, reuse=self.reuse, biased=False, relu=False,
                 name=name + 'upscore2'))

        (self.feed(name + 'upscore2', name + 'score_pool4')
         .add(name=name + 'fuse_pool4')
         .deconv([4, 4], pool3_shape, [2, 2], num_classes, reuse=self.reuse, biased=False, relu=False,
                 name=name + 'upscore_pool4'))

        origin_shape = tf.multiply(tf.shape(self.layers['data']), tf.convert_to_tensor([1, 1, 1, 7]))
        (self.feed(name + 'upscore_pool4', name + 'score_pool3')
         .add(name=name + 'fuse_pool3')
         .deconv([16, 16], origin_shape, [8, 8], num_classes, reuse=self.reuse, biased=False, relu=False,
                 name=name + 'upscore8'))

    def topredict(self, raw_output, origin_shape=None):
        # raw_output = tf.image.resize_bilinear(raw_output, origin_shape)
        raw_output = tf.argmax(raw_output, axis=3)
        prediction = tf.expand_dims(raw_output, dim=3)
        return prediction
