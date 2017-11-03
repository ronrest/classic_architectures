import tensorflow as tf

# Convenient layer operation shortcuts
fc = tf.contrib.layers.fully_connected
conv = tf.contrib.layers.conv2d
avgpool = tf.contrib.layers.avg_pool2d
dropout_layer = tf.layers.dropout
batchnorm = tf.contrib.layers.batch_norm
winit = tf.contrib.layers.xavier_initializer()
arg_scope = tf.contrib.framework.arg_scope


# ==============================================================================
#                                                            GET_LENET5_ARGSCOPE
# ==============================================================================
def get_lenet5_argscope(weight_decay=None, use_batch_norm=False, is_training=False):
    """ Gets the arg scope needed for Lenet5.
    Args:
        weight_decay: The l2 regularization coefficient.
    Returns:
        An arg_scope with the default arguments for layers in lenet5 .
    """
    with tf.contrib.framework.arg_scope(
        [conv, fc],
        activation_fn=tf.nn.sigmoid,
        normalizer_fn = batchnorm if use_batch_norm else None,
        normalizer_params = {"is_training": is_training},
        weights_regularizer=None if weight_decay is None else l2_regularizer(weight_decay),
        weights_initializer=winit,
        biases_initializer=tf.zeros_initializer(),
        trainable = True):
        with tf.contrib.framework.arg_scope([conv, avgpool], padding='VALID') as scope:
            return scope


# ==============================================================================
#                                                                         LENET5
# ==============================================================================
def lenet5(X, n_classes=10, is_training=False):
    """ From Lecun et al 1998, section II.

        PARAMETERS: 61,706

    Notes:
        Original architecture was designed to be used on 32x32 images.

    References:
        Y. LeCun, L. Bottou, Y. Bengio and P. Haffner: Gradient-Based
            Learning Applied to Document Recognition, Proceedings of the
            IEEE, 86(11):2278-2324, November 1998
            http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
    """
    with tf.name_scope("preprocessing"):
        x = tf.div(X, 255., name="scaled_inputs")

    with tf.variable_scope("lenet5"):
        with arg_scope(get_lenet5_argscope(use_batch_norm=False, is_training=is_training)):
            x = conv(x, num_outputs=6, kernel_size=5, scope="conv1")
            with variable_scope("avgpool1"):
                x = avgpool(x, kernel_size=2, stride=2, scope="pool")
                x = tf.nn.sigmoid(x, name="sigmoid")

            x = conv(x, num_outputs=16, kernel_size=5, scope="conv2")
            with variable_scope("avgpool2"):
                x = avgpool(x, kernel_size=2, stride=2, scope="pool")
                x = tf.nn.sigmoid(x, name="sigmoid")

            x = tf.contrib.layers.flatten(x)
            x = fc(x, num_outputs=120, scope="fc3")
            x = fc(x, num_outputs=84, scope="fc4")
            x = fc(x, num_outputs=n_classes, activation_fn=None, scope="fc5")
    return x
