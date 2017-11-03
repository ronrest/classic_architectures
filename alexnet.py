import tensorflow as tf

# Convenient layer operation shortcuts
fc = tf.contrib.layers.fully_connected
conv = tf.contrib.layers.conv2d
# convsep = tf.contrib.layers.separable_conv2d
relu = tf.nn.relu
maxpool = tf.contrib.layers.max_pool2d
dropout_layer = tf.layers.dropout
batchnorm = tf.contrib.layers.batch_norm
winit = tf.contrib.layers.xavier_initializer()
arg_scope = tf.contrib.framework.arg_scope


# ==============================================================================
#                                                           GET_ALEXNET_ARGSCOPE
# ==============================================================================
def get_alexnet_argscope(weight_decay=None, use_batch_norm=False, is_training=False):
    """ Gets the arg scope needed for Alexnet.
    Args:
        weight_decay: The l2 regularization coefficient.
    Returns:
        An arg_scope with the default arguments for layers.
    """
    with tf.contrib.framework.arg_scope(
        [conv, fc],
        activation_fn=tf.nn.relu,
        normalizer_fn = batchnorm if use_batch_norm else None,
        normalizer_params = {"is_training": is_training},
        weights_regularizer=None if weight_decay is None else l2_regularizer(weight_decay),
        weights_initializer=winit,
        biases_initializer=tf.zeros_initializer(),
        trainable = True):
        with tf.contrib.framework.arg_scope([conv], padding='SAME') as scope:
            return scope


# ==============================================================================
#                                                                        ALEXNET
# ==============================================================================
def alexnet(X, n_classes=10, is_training=False):
    """ Krizhevsky et al 2012.

    PARAMETERS: 62,378,344 (for input of [127x127x3], and 1000 output classes)

    Notes:
        Input images in original are 227x227x3 (incorrectly mentioned in paper
        as 224x224x3)

        Input images should be at least 67x67 for this architecture to work.

    References:
        Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, 2012:
            ImageNet Classification with Deep Convolutional Neural Networks
            https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

    TODO:
        This implementation is not making use of local response normalization.
    """
    with tf.name_scope("preprocessing"):
        x = tf.div(X, 255., name="scaled_inputs")

    with tf.variable_scope("alexnet"):
        with arg_scope(get_alexnet_argscope(weight_decay=None, use_batch_norm=False, is_training=is_training)):
            x = conv(x, num_outputs=96, kernel_size=11, stride=4, padding="VALID", scope="conv1")
            x = maxpool(x, kernel_size=3, stride=2, scope="maxpool1")

            x = conv(x, num_outputs=256, kernel_size=5, scope="conv2")
            x = maxpool(x, kernel_size=3, stride=2, scope="maxpool2")

            x = conv(x, num_outputs=384, kernel_size=3, scope="conv3")
            x = conv(x, num_outputs=384, kernel_size=3, scope="conv4")
            x = conv(x, num_outputs=256, kernel_size=3, scope="conv5")
            x = maxpool(x, kernel_size=3, stride=2, scope="maxpool5")
            print("Final maxplool", x.shape.as_list())

            x = tf.contrib.layers.flatten(x)
            print("Flattened", x.shape.as_list())
            x = fc(x, num_outputs=4096, scope="fc6")
            x = fc(x, num_outputs=4096, scope="fc7")
            x = fc(x, num_outputs=n_classes, activation_fn=None, scope="fc8")
    return x
