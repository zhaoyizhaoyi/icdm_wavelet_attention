import tensorflow as tf
import logging
from tensorflow.contrib.layers import flatten
from .general_utils import batch_normalization


# configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VGG(object):
    def __init__(self, training, keep_prob=1.0, num_layers=19,
                 for_attention=False, initial_filters=64, kernel_size=3):
        self.training = training
        self.keep_prob = keep_prob
        self.num_layers = num_layers
        self.for_attention = for_attention
        self.initial_filters = initial_filters
        self.kernel_size = kernel_size

    def _conv_block(self, block_num, inputs_x, filters, num_layers, pool_size=2, ignore_pooling=False):
        for i in range(num_layers):
            with tf.name_scope("conv_" + str(block_num) + "_" + str(i)):
                inputs_x = tf.layers.conv2d(
                    inputs=inputs_x,
                    filters=filters,
                    kernel_size=self.kernel_size,
                    strides=1,
                    padding="same",
                    activation=tf.nn.relu
                    # activation=tf.nn.leaky_relu
                    # activation=tf.nn.selu
                )
        if ignore_pooling:
            return inputs_x
        else:
            with tf.name_scope("max_pool_" + str(block_num)):
                max_pool_x = tf.layers.max_pooling2d(
                    inputs=inputs_x,
                    pool_size=pool_size,
                    strides=2,
                    padding="same"
                )
            logger.info("max_pool_{}: {}".format(block_num, max_pool_x.shape))

            return max_pool_x

    def build_2_layers(self, inputs, add_bn=False):
        max_pool_1 = self._conv_block(
            block_num=1,
            inputs_x=inputs,
            filters=self.initial_filters,
            num_layers=2,
            pool_size=2,
            ignore_pooling=True if self.for_attention else False
        )

        if add_bn:
            max_pool_1 = batch_normalization(max_pool_1, training=self.training, scope="batch_1")

        return max_pool_1

    def build_4_layers(self, inputs, add_bn=False):
        max_pool_1 = self._conv_block(
            block_num=1,
            inputs_x=inputs,
            filters=self.initial_filters,
            num_layers=2,
            pool_size=2
        )

        # block 2
        max_pool_2 = self._conv_block(
            block_num=2,
            inputs_x=max_pool_1,
            filters=self.initial_filters * 2,
            num_layers=2,
            pool_size=2,
            ignore_pooling=True if self.for_attention else False
        )

        if add_bn:
            max_pool_2 = batch_normalization(max_pool_2, training=self.training, scope="batch_5")

        return max_pool_2

    def build_6_layers(self, inputs, add_bn=False):
        max_pool_1 = self._conv_block(
            block_num=1,
            inputs_x=inputs,
            filters=self.initial_filters,
            num_layers=2
        )

        # block 2
        max_pool_2 = self._conv_block(
            block_num=2,
            inputs_x=max_pool_1,
            filters=self.initial_filters * 2,
            num_layers=2
        )

        # block 3
        max_pool_3 = self._conv_block(
            block_num=3,
            inputs_x=max_pool_2,
            filters=self.initial_filters * 4,
            num_layers=2,
            ignore_pooling=True if self.for_attention else False
        )

        if add_bn:
            max_pool_3 = batch_normalization(max_pool_3, training=self.training, scope="batch_5")

        return max_pool_3

    def build_11_layers(self, inputs, add_bn=False):
        # block 1
        max_pool_1 = self._conv_block(
            block_num=1,
            inputs_x=inputs,
            filters=self.initial_filters,
            num_layers=2,
            pool_size=2
        )

        # block 2
        max_pool_2 = self._conv_block(
            block_num=2,
            inputs_x=max_pool_1,
            filters=self.initial_filters * 2,
            num_layers=2,
            pool_size=2
        )

        # block 3
        max_pool_3 = self._conv_block(
            block_num=3,
            inputs_x=max_pool_2,
            filters=self.initial_filters * 4,
            num_layers=3,
            pool_size=2
        )

        # block 4
        max_pool_4 = self._conv_block(
            block_num=4,
            inputs_x=max_pool_3,
            filters=self.initial_filters * 8,
            num_layers=3,
            pool_size=2,
            ignore_pooling=True if self.for_attention else False
        )

        if add_bn:
            max_pool_4 = batch_normalization(max_pool_4, training=self.training, scope="batch_5")

        return max_pool_4

    def build_16_layers(self, inputs, add_bn=False):
        # block 1
        max_pool_1 = self._conv_block(
            block_num=1,
            inputs_x=inputs,
            filters=self.initial_filters,
            num_layers=2
        )

        # block 2
        max_pool_2 = self._conv_block(
            block_num=2,
            inputs_x=max_pool_1,
            filters=self.initial_filters * 2,
            num_layers=2
        )

        # block 3
        max_pool_3 = self._conv_block(
            block_num=3,
            inputs_x=max_pool_2,
            filters=self.initial_filters * 4,
            num_layers=3
        )

        # block 4
        max_pool_4 = self._conv_block(
            block_num=4,
            inputs_x=max_pool_3,
            filters=self.initial_filters * 8,
            num_layers=3
        )

        # block 5
        max_pool_5 = self._conv_block(
            block_num=5,
            inputs_x=max_pool_4,
            filters=self.initial_filters * 8,
            num_layers=3,
            ignore_pooling=True if self.for_attention else False
        )

        if add_bn:
            max_pool_5 = batch_normalization(max_pool_5, training=self.training, scope="batch_5")

        return max_pool_5

    def build_19_layers(self, inputs, add_bn=False):
        # block 1
        max_pool_1 = self._conv_block(
            block_num=1,
            inputs_x=inputs,
            filters=self.initial_filters,
            num_layers=2
        )

        # block 2
        max_pool_2 = self._conv_block(
            block_num=2,
            inputs_x=max_pool_1,
            filters=self.initial_filters * 2,
            num_layers=2
        )

        # block 3
        max_pool_3 = self._conv_block(
            block_num=3,
            inputs_x=max_pool_2,
            filters=self.initial_filters * 4,
            num_layers=4
        )

        # block 4
        max_pool_4 = self._conv_block(
            block_num=4,
            inputs_x=max_pool_3,
            filters=self.initial_filters * 8,
            num_layers=4
        )

        # block 5
        max_pool_5 = self._conv_block(
            block_num=5,
            inputs_x=max_pool_4,
            filters=self.initial_filters * 8,
            num_layers=4,
            ignore_pooling=True if self.for_attention else False
        )

        if add_bn:
            max_pool_5 = batch_normalization(max_pool_5, training=self.training, scope="batch_5")

        return max_pool_5

    def build_vgg(self, inputs):
        logger.info("Building {}-layer VGG..".format(self.num_layers))
        if self.num_layers == 2:
            flat = self.build_2_layers(inputs)
        elif self.num_layers == 4:
            flat = self.build_4_layers(inputs)
        elif self.num_layers == 6:
            flat = self.build_6_layers(inputs)
        elif self.num_layers == 11:
            flat = self.build_11_layers(inputs)
        elif self.num_layers == 16:
            flat = self.build_16_layers(inputs)
        elif self.num_layers == 19:
            flat = self.build_19_layers(inputs)
        else:
            raise ValueError("Not valid vgg_num_layers, choose from [2, 4, 6, 11, 16, 19]!")

        # flatten and dropout
        if not self.for_attention:
            flat = flatten(flat)
        logger.info("flat shape: {}".format(flat.shape))

        if self.keep_prob is not None:
            flat = tf.layers.dropout(inputs=flat, rate=1.0-self.keep_prob, training=self.training)

        return flat

    def build_vgg_with_bn(self, inputs):
        if self.num_layers == 2:
            flat = self.build_2_layers(inputs, add_bn=True)
        elif self.num_layers == 4:
            flat = self.build_4_layers(inputs, add_bn=True)
        elif self.num_layers == 6:
            flat = self.build_6_layers(inputs, add_bn=True)
        elif self.num_layers == 11:
            flat = self.build_11_layers(inputs, add_bn=True)
        elif self.num_layers == 16:
            flat = self.build_16_layers(inputs, add_bn=True)
        elif self.num_layers == 19:
            flat = self.build_19_layers(inputs, add_bn=True)
        else:
            raise ValueError("Not valid vgg_num_layers, choose from [2, 4, 6, 11, 16, 19]!")

        return flat

