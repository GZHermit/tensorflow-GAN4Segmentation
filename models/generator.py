# coding:utf-8
from models.network import NetWork
import tensorflow as tf


class Generator(NetWork):
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
         .conv([3, 3], 512, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv5_3'))

        (self.feed(name + 'image_conv5_3')
         .conv([1, 1], num_classes, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'conv_final'))

    def topredict(self, raw_output, origin_shape):
        raw_output = tf.image.resize_bilinear(raw_output, origin_shape)
        raw_output = tf.argmax(raw_output, dimension=3)
        prediction = tf.expand_dims(raw_output, dim=3)
        return prediction
