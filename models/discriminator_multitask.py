from models.network import NetWork


def choose_discriminator(d_name, fk_batch, gt_batch, image_batch):
    if d_name == 'disc_vgg16':
        d_fk_net = Discriminator_vgg16({'seg': fk_batch})
        d_gt_net = Discriminator_vgg16({'seg': gt_batch}, reuse=True)
    elif d_name == 'disc_resnet50':
        d_fk_net = Discriminator_resnet50({'seg': fk_batch, 'data': image_batch})
        d_gt_net = Discriminator_resnet50({'seg': gt_batch, 'data': image_batch}, reuse=True)
    return d_fk_net, d_gt_net


class Discriminator_vgg16(NetWork):
    def setup(self, *args):
        name = 'discriminator_vgg16/'
        (self.feed('seg')
         .conv([5, 5], 64, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'seg_conv_1'))

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


class Discriminator_resnet50(NetWork):
    def setup(self, *args):
        pass
