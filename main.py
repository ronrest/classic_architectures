from __future__ import print_function, division
import numpy as np
import tensorflow as tf

# Import the architectures
from vgg16 import vgg16
from lenet5 import lenet5
from alexnet import alexnet

# ==============================================================================
#                                                            PRINT_MODEL_SUMMARY
# ==============================================================================
def print_model_summary(graph):
    print("MODEL PARAMETERS")
    template = "- {name:<30}:  {params: 8d} parameters. {shape}"
    total_params = 0
    with graph.as_default():
        vars = tf.trainable_variables()
        for var in vars:
            shape = var.shape.as_list()
            n_params = np.prod(shape)
            total_params += n_params
            print(template.format(name=var.name, params=n_params, shape=shape))
    print("- TOTAL PARAMETERS:", total_params)


################################################################################
#  CREATE THE GRAPH
################################################################################
graph = tf.Graph()
with graph.as_default():
    n_classes = 1000
    tf_X = tf.placeholder(tf.float32, shape=(None,224,224,3))
    tf_Y = tf.placeholder(tf.float32, shape=(None,n_classes))
    is_training = tf.placeholder_with_default(False, shape=1)
    tf_logits = vgg16(tf_X, n_classes=n_classes, is_training=is_training)
    print_model_summary(graph=graph)

    # -------------------
    # TODO: Add your own losses and optimization operations
    # -------------------
    # tf_loss = ...
    # tf_trainstep = ...


################################################################################
#  RUN SESSION
################################################################################
# -------------------
# TODO: Create your own training loop
# -------------------
# with tf.Session(graph=graph) as sess:
#     sess.run(tf.global_variables_initializer())
#     X = np.random.randn(10, 224, 224,3)
#     Y = np.random.randn(10, n_classes)
#     loss = sess.run([tf_loss, tf_trainstep], feed_dict={tf_X: X, tf_Y:Y, is_training: True})
#     print(loss)
