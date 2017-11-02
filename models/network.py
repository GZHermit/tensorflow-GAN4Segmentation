# coding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from math import ceil


def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the inpu for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class NetWork(object):
    def __init__(self, inputs, reuse=False, trainable=True, is_training=False, num_classes=21):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Switch variable for dropout
        self.use_dropout = tf.placeholder_with_default(tf.constant(0.8, dtype=tf.float32),
                                                       shape=[], name='use_dropout')
        # If true, the D-model's convlayer can be reused.
        self.reuse = reuse

        self.setup(is_training, num_classes)

    def setup(self, *args):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path).item()
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_appointed_layer(self, name):
        return self.layers[name]

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        '''Creates a new TensorFlow variable.'''
        return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=0.03),
                               trainable=self.trainable)

    def make_deconv_filter(self, name, filter_shape):
        width, heigh = filter_shape[0], filter_shape[1]
        f = ceil(width / 2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([filter_shape[0], filter_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(filter_shape)
        for i in range(filter_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights, dtype=tf.float32)
        return tf.get_variable(name=name, initializer=init, shape=weights.shape)

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self, input, kernel, output_channel, strides, name, reuse=None,
             relu=True,
             padding='SAME',
             group=1,
             biased=True):

        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        input_channel = input.get_shape().as_list()[-1]
        # Verify that the grouping parameter is valid
        assert input_channel % group == 0
        assert output_channel % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, strides[0], strides[1], 1], padding=padding)
        with tf.variable_scope(name, reuse=reuse) as scope:
            filter = self.make_var('weights', shape=[kernel[0], kernel[1], input_channel / group, output_channel])
            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(input, filter)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(input, group, 3)
                kernel_groups = tf.split(filter, group, 3)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(output_groups, 3)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [output_channel])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def atrous_conv(self, input, kernel, output_channel, dilation, name, reuse=None,
                    relu=True,
                    padding='SAME',
                    group=1,
                    biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        input_channel = input.get_shape().as_list()[-1]
        # Verify that the grouping parameter is valid
        assert input_channel % group == 0
        assert output_channel % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.atrous_conv2d(i, k, dilation, padding=padding)
        with tf.variable_scope(name, reuse=reuse) as scope:
            filter = self.make_var('weights', shape=[kernel[0], kernel[1], input_channel / group, output_channel])
            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(input, filter)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(input, group, 3)
                kernel_groups = tf.split(filter, group, 3)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(output_groups, 3)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [output_channel])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def deconv(self, input, kernel, output_shape, strides, output_channel, name, reuse=None,
               relu=False,
               padding='SAME',
               biased=False):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        input_channel = input.get_shape().as_list()[-1]

        deconvolve = lambda i, k, os: tf.nn.conv2d_transpose(i, k, os, [1, strides[0], strides[1], 1],
                                                             padding=padding)
        with tf.variable_scope(name, reuse=reuse) as scope:

            output_shape = [output_shape[0], output_shape[1], output_shape[2], output_channel]
            f_shape = [kernel[0], kernel[1], output_channel, input_channel]
            filter = self.make_deconv_filter('weights', f_shape)
            output = deconvolve(input, filter, output_shape)
            if biased:
                biases = self.make_var('biases', [output_channel])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
        return output

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def sigmoid(self, input, name):
        return tf.nn.sigmoid(input, name=name)

    @layer
    def max_pool(self, input, kernel, strides, name, padding='SAME'):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, kernel[0], kernel[1], 1],
                              strides=[1, strides[0], strides[1], 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, kernel, strides, name, padding='SAME'):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, kernel[0], kernel[1], 1],
                              strides=[1, strides[0], strides[1], 1],
                              padding=padding,
                              name=name)

    @layer
    def global_average_pooling(self, input, name):
        return tf.reduce_mean(input, axis=[1, 2], keep_dims=True, name=name)

    # @layer
    # 不用这个是因为在validate的时候，图片尺寸不再固定，而avg_pool里的ksize要求是the list of int
    # def global_average_pooling(self, input, name):
    #     ksize = [1, ] + input.get_shape().as_list()[1:3] + [1, ]
    #     return tf.nn.avg_pool(input,
    #                           ksize=ksize,
    #                           strides=[1, 1, 1, 1],
    #                           padding='VALID',
    #                           name=name)

    @layer
    def lrn(self, input, name, radius=None, alpha=None, beta=None, bias=None):
        return tf.nn.lrn(input,
                         depth_radius=radius,
                         alpha=alpha,
                         beta=beta,
                         bias=bias,
                         name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(values=inputs, axis=axis, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, input, num_out, name, reuse=False, relu=True):
        with tf.variable_scope(name, reuse=reuse) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = map(lambda v: v.value, input.get_shape())
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                input = tf.squeeze(input, squeeze_dims=[1, 2])
            else:
                raise ValueError('Rank 2 tensor input expected for softmax!')
        return tf.nn.softmax(input, name)

    @layer
    def batch_normalization(self, input, name, is_training, reuse=None, activation_fn=None, scale=True):
        with tf.variable_scope(name, reuse=reuse) as scope:
            output = slim.batch_norm(
                input,
                decay=0.9997,
                activation_fn=activation_fn,
                is_training=is_training,
                updates_collections=None,
                scale=scale,
                scope=scope)
            return output

    @layer
    def dropout(self, input, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        return tf.nn.dropout(input, keep, name=name)

    @layer
    def resize(self, input, new_size, name):
        return tf.image.resize_bilinear(input, new_size, name=name)

    @layer
    def resize_nn(self, input, new_size, name):
        return tf.image.resize_nearest_neighbor(input, new_size, name=name)
