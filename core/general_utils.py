import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope


def batch_normalization(x, scope, training=True):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(
            training,
            lambda: batch_norm(inputs=x, is_training=training, reuse=None),
            lambda: batch_norm(inputs=x, is_training=training, reuse=True)
        )


