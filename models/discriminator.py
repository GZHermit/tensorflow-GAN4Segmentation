from models.network import NetWork

def choose_discriminator(d_name, fk_batch, gt_batch, image_batch):
    if d_name == 'disc':
        d_fk_net = Discriminator({'seg': fk_batch})
        d_gt_net = Discriminator({'seg': gt_batch}, reuse=True)
    elif d_name == 'disc_addx':
        d_fk_net = Discriminator_addx({'seg': fk_batch, 'data': image_batch})
        d_gt_net = Discriminator_addx({'seg': gt_batch, 'data': image_batch}, reuse=True)
    elif d_name == 'disc_add_vgg':
        d_fk_net = Discriminator_add_vgg({'seg': fk_batch, 'data': image_batch})
        d_gt_net = Discriminator_add_vgg({'seg': gt_batch, 'data': image_batch}, reuse=True)
    return d_fk_net, d_gt_net
class Discriminator(NetWork):
    def setup(self, *args):
        name = 'discriminator/'
        (self.feed('seg')
         .conv([3, 3], 96, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'conv_1')
         .conv([3, 3], 128, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'conv_2')
         .conv([3, 3], 128, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'conv_3')
         .max_pool([2, 2], [2, 2], name=name + 'maxpool1')
         .conv([3, 3], 256, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'conv_4')
         .conv([3, 3], 256, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'conv_5')
         .max_pool([2, 2], [2, 2], name=name + 'maxpool2')
         .conv([3, 3], 512, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'conv_6')
         .conv([3, 3], 2, [1, 1], reuse=self.reuse, biased=True, relu=False, name=name + 'conv_7'))

        (self.feed(name + 'conv_7')
         .fc(1, reuse=self.reuse, relu=False, name=name + 'fc')
         .sigmoid(name=name + 'sigmoid'))


class Discriminator_addx(NetWork):
    def setup(self, *args):
        name = 'discriminator_addx/'
        (self.feed('seg')
         .conv([5, 5], 64, [4, 4], reuse=self.reuse, biased=True, relu=True, name=name + 'seg_conv_1'))

        (self.feed('data')
         .conv([5, 5], 16, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv_1')
         .max_pool([2, 2], [2, 2], name=name + 'image_maxpool1')
         .conv([5, 5], 64, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv_2')
         .max_pool([2, 2], [2, 2], name=name + 'image_maxpool2'))

        (self.feed(name + 'seg_conv_1',
                   name + 'image_maxpool2')
         .concat(axis=3, name=name + 'concat')
         .conv([3, 3], 128, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'conv_1')
         .max_pool([2, 2], [2, 2], name=name + 'maxpool1')
         .conv([3, 3], 256, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'conv_2')
         .max_pool([2, 2], [2, 2], name=name + 'maxpool2')
         .conv([3, 3], 512, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'conv_3')
         .conv([3, 3], 2, [1, 1], reuse=self.reuse, biased=True, relu=False, name=name + 'conv_4'))

        (self.feed(name + 'conv_4')
         .fc(1, reuse=self.reuse, relu=False, name=name + 'fc')
         .sigmoid(name=name + 'sigmoid'))


class Discriminator_add_vgg(NetWork):
    def setup(self, *args):
        name = 'discriminator_add_vgg/'
        (self.feed('seg')
         .conv([5, 5], 64, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'seg_conv_1'))

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
         .conv([3, 3], 64, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'conv5_4')
         )

        (self.feed(name + 'seg_conv_1',
                   name + 'conv5_4')
         .concat(axis=3, name=name + 'concat')
         .conv([3, 3], 128, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'conv_1')
         .max_pool([2, 2], [2, 2], name=name + 'maxpool1')
         .conv([3, 3], 256, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'conv_2')
         .max_pool([2, 2], [2, 2], name=name + 'maxpool2')
         .conv([3, 3], 512, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'conv_3')
         .conv([3, 3], 2, [1, 1], reuse=self.reuse, biased=True, relu=False, name=name + 'conv_4'))

        (self.feed(name + 'conv_4')
         .fc(1, reuse=self.reuse, relu=False, name=name + 'fc')
         .sigmoid(name=name + 'sigmoid'))
