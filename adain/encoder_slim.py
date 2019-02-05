import tensorflow.contrib.slim as slim


import tensorflow as tf
import numpy as np
class Encoder(object):
    def __init__(self, input_tensor):
        self.input = input_tensor

        x = input_tensor
        x = tf.reverse(x, axis=[-1])
        x = x - tf.constant(np.array([103.939, 116.779, 123.68], dtype=np.float32))

        # block1
        x = tf.pad(x, tf.constant([[0,0], [1,1], [1,1], [0,0]]), mode="REFLECT")
        x = slim.conv2d(x, 64, [3, 3], padding='VALID', scope='block1_conv1')
        x = tf.pad(x, tf.constant([[0,0], [1,1], [1,1], [0,0]]), mode="REFLECT")
        x = slim.conv2d(x, 64, [3, 3], padding='VALID', scope='block1_conv2')
        x = slim.max_pool2d(x, [2, 2], scope='block1_pool')

        # block2
        x = tf.pad(x, tf.constant([[0,0], [1,1], [1,1], [0,0]]), mode="REFLECT")
        x = slim.conv2d(x, 128, [3, 3], padding='VALID', scope='block2_conv1')
        x = tf.pad(x, tf.constant([[0,0], [1,1], [1,1], [0,0]]), mode="REFLECT")
        x = slim.conv2d(x, 128, [3, 3], padding='VALID', scope='block2_conv2')
        x = slim.max_pool2d(x, [2, 2], scope='block2_pool')

        # block3
        x = tf.pad(x, tf.constant([[0,0], [1,1], [1,1], [0,0]]), mode="REFLECT")
        x = slim.conv2d(x, 256, [3, 3], padding='VALID', scope='block3_conv1')
        x = tf.pad(x, tf.constant([[0,0], [1,1], [1,1], [0,0]]), mode="REFLECT")
        x = slim.conv2d(x, 256, [3, 3], padding='VALID', scope='block3_conv2')
        x = tf.pad(x, tf.constant([[0,0], [1,1], [1,1], [0,0]]), mode="REFLECT")
        x = slim.conv2d(x, 256, [3, 3], padding='VALID', scope='block3_conv3')
        x = tf.pad(x, tf.constant([[0,0], [1,1], [1,1], [0,0]]), mode="REFLECT")
        x = slim.conv2d(x, 256, [3, 3], padding='VALID', scope='block3_conv4')
        x = slim.max_pool2d(x, [2, 2], scope='block3_pool1')

        # block4
        x = tf.pad(x, tf.constant([[0,0], [1,1], [1,1], [0,0]]), mode="REFLECT")
        x = slim.conv2d(x, 512, [3, 3], padding='VALID', scope='block4_conv1')
        self.output = x

    def load_ckpt(self, sess, ckpt='ckpts/vgg_16.ckpt'):
        variables = slim.get_variables(scope='vgg_16')
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(ckpt, variables)
        sess.run(init_assign_op, init_feed_dict)

    def get_activation(self, layer_name='conv5_1'):
        return self.layers[layer_name]


if __name__ == '__main__':
    x_pl = tf.placeholder(tf.float32, [None, 256, 256, 3])
    encoder = Encoder(x_pl)
    
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        features = sess.run(encoder.output, {x_pl: np.zeros((1,256,256,3))})
        print(features.shape)



