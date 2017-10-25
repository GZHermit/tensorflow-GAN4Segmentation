from models.network import NetWork
import tensorflow as tf


def choose_discriminator(d_name, fk_batch, gt_batch, feature_batch):
    if d_name == 'disc_vgg16':
        d_fk_net = Discriminator_add_vgg({'seg': fk_batch, 'data': feature_batch})
        d_gt_net = Discriminator_add_vgg({'seg': gt_batch, 'data': feature_batch}, reuse=True)
    elif d_name == 'disc_resnet50':
        d_fk_net = Discriminator_add_res50({'seg': fk_batch, 'data': feature_batch})
        d_gt_net = Discriminator_add_res50({'seg': gt_batch, 'data': feature_batch}, reuse=True)
    return d_fk_net, d_gt_net


class Discriminator_add_vgg(NetWork):
    def setup(self, *args):
        name = 'discriminator_add_vgg/'

        (self.feed('data')
         .conv([3, 3], 64, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'conv5_4'))
        feature_shape = tf.shape(self.layers[name + 'conv5_4'])

        (self.feed('seg')
         .resize_nn(feature_shape[1:3], name=name + 'resize_nn')
         .conv([1, 1], 64, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'seg_conv_1'))

        (self.feed(name + 'seg_conv_1',
                   name + 'conv5_4')
         .concat(axis=3, name=name + 'concat')
         .conv([3, 3], 128, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'conv_1')
         .max_pool([2, 2], [2, 2], name=name + 'maxpool1')
         .conv([3, 3], 256, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'conv_2')
         .max_pool([2, 2], [2, 2], name=name + 'maxpool2')
         .conv([3, 3], 512, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'conv_3')
         .conv([3, 3], 1, [1, 1], reuse=self.reuse, biased=True, relu=False, name=name + 'conv_4'))


class Discriminator_add_res50(NetWork):
    def setup(self, *args):
        name = 'discriminator_add_res50/'
        pass
