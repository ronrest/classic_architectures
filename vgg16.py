"""
Based on this code:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/vgg.py

But allows for the inclusion of batchnorm for each convolution layer.
"""
import tensorflow as tf

# USEFUL LAYERS
fc = tf.contrib.layers.fully_connected
conv = tf.contrib.layers.conv2d
# convsep = tf.contrib.layers.separable_conv2d
deconv = tf.contrib.layers.conv2d_transpose
relu = tf.nn.relu
maxpool = tf.contrib.layers.max_pool2d
dropout_layer = tf.layers.dropout
batchnorm = tf.contrib.layers.batch_norm
winit = tf.contrib.layers.xavier_initializer()
l2_regularizer = tf.contrib.layers.l2_regularizer
repeat = tf.contrib.layers.repeat
arg_scope = tf.contrib.framework.arg_scope

# ==============================================================================
#                                                               GET_VGG_ARGSCOPE
# ==============================================================================
def get_vgg_argscope(weight_decay=0.0005, use_batch_norm=False, is_training=False):
    """ Gets the arg scope needed for VGG.
    Args:
        weight_decay: The l2 regularization coefficient.
    Returns:
        An arg_scope with the default arguments for layers in VGG
    Credits:
        Based on this code:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/vgg.py
    """
    with tf.contrib.framework.arg_scope(
        [conv],
        activation_fn=tf.nn.relu,
        normalizer_fn = batchnorm if use_batch_norm else None,
        normalizer_params = {"is_training": is_training},
        weights_regularizer=l2_regularizer(weight_decay),
        biases_initializer=tf.zeros_initializer(),
        trainable = True):
        with tf.contrib.framework.arg_scope([conv], padding='SAME') as scope:
                return scope


# ==============================================================================
#                                                                          VGG16
# ==============================================================================
def vgg16(inputs, n_classes=1000, use_batch_norm=False, is_training=True, dropout=0.5, weight_decay=0.0005, spatial_squeeze=True):
    """ Simonyan and Zisserman 2014

    PARAMETERS: 138,357,544 (with input image [224,224,3] and 1000 classes)
    PARAMETERS IN CONV LAYERS: 14,714,688

    Notes:
        Originally designed for 224x224x3 images.

    References:
        Karen Simonyan, Andrew Zisserman, 2014: Very Deep Convolutional
        Networks for Large-Scale Image Recognition
        https://arxiv.org/abs/1409.1556
    Credits:
        Based on this code:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/vgg.py
    """
    with tf.variable_scope("vgg16", "vgg16"):
        # Trunk of convolutional layers
        with tf.contrib.framework.arg_scope(get_vgg_argscope(
                weight_decay=weight_decay,
                use_batch_norm=use_batch_norm,
                is_training=is_training)):
            x = repeat(inputs, 2, conv, num_outputs=64, kernel_size=3, scope='conv1')
            x = maxpool(x, kernel_size=2, scope='pool1')
            x = repeat(x, 2, conv, num_outputs=128, kernel_size=3, scope='conv2')
            x = maxpool(x, kernel_size=2, scope='pool2')
            x = repeat(x, 3, conv, num_outputs=256, kernel_size=3, scope='conv3')
            x = maxpool(x, kernel_size=2, scope='pool3')
            x = repeat(x, 3, conv, num_outputs=512, kernel_size=3, scope='conv4')
            x = maxpool(x, kernel_size=2, scope='pool4')
            x = repeat(x, 3, conv, num_outputs=512, kernel_size=3, scope='conv5')
            x = maxpool(x, kernel_size=2, scope='pool5')

            # "Fully connected" layers
            # Use conv2d instead of fully_connected layers.
            x = conv(x, num_outputs=4096, kernel_size=7, padding='VALID', scope='fc6')
            x = dropout_layer(x, dropout, training=is_training, name='dropout6')
            x = conv(x, num_outputs=4096, kernel_size=1, scope='fc7')
            x = dropout_layer(x, dropout, training=is_training, name='dropout7')
            x = conv(x, num_outputs=n_classes, kernel_size=1, activation_fn=None, normalizer_fn=None, scope='fc8')
            if spatial_squeeze:
                x = tf.squeeze(x, [1, 2], name='fc8/squeezed')
        return x
