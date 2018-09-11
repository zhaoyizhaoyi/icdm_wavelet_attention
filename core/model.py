from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
import logging
import random
import time
import argparse
import math
from sklearn.metrics import log_loss, accuracy_score, mean_squared_error
from .vgg import VGG


# configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WaveletAttentionNetwork(object):
    """
    Build a CNN on time series to predict future values.
    """
    def __init__(self, sess, ahead_step, batch_size, learning_rate,
                 keep_prob,
                 time_window,
                 num_channels, num_classes,
                 max_training_iters=50,
                 display_step=100,
                 model_structure=1,
                 lstm_units=10,
                 lstm_num_layers=1,
                 two_dense_layers=False,
                 decay_dense_net=False,
                 restore_to_test=True,
                 vgg_num_layers=19,
                 dense_units=1024,
                 vgg_initial_filters=64,
                 training_slice=0,
                 num_wavelet_channels=2,
                 add_l2=False,
                 weight_decay=0.0001,
                 lstm_name="lstm",
                 ensemble_lstm=0.5,
                 export_attention_weights=False,
                 model_timestamp=int(time.time() * 1000)):
        self.sess = sess
        self.ahead_step = ahead_step
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.time_window = time_window
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.max_training_iters = max_training_iters
        self.display_step = display_step
        self.model_structure = model_structure
        self.lstm_units = lstm_units
        self.lstm_num_layers = lstm_num_layers
        self.two_dense_layers = two_dense_layers
        self.decay_dense_net = decay_dense_net
        self.restore_to_test = restore_to_test
        self.vgg_num_layers = vgg_num_layers
        self.dense_units = dense_units
        self.vgg_initial_filters = vgg_initial_filters
        self.training_slice = training_slice
        self.num_wavelet_channels = num_wavelet_channels
        self.add_l2 = add_l2
        self.weight_decay = weight_decay
        self.lstm_name = lstm_name
        self.ensemble_lstm = ensemble_lstm
        self.export_attention_weights = export_attention_weights
        self.model_timestamp = model_timestamp

        self.inputs = None
        self.pure_lstm_inputs = None
        self.labels = None
        self.pred = None
        self.loss = None
        self.optim = None
        self.accuracy = None
        self.merged_summary_op = None
        self.training_flag = True
        self.dropout_ratio_placeholder = None
        self.alpha_weights = None

        if self.model_structure == 1:
            self.build_graph_pure_lstm()
        elif self.model_structure == 2:
            self.build_graph_vgg()
        elif self.model_structure == 3:
            self.build_graph_attention_vgg_lstm()
        elif self.model_structure == 4:
            self.build_graph_ensemble()
        else:
            logger.info("Please specify the right model_structure number (choose from 1, 2, 3, 4)")
            exit(1)

    def _create_one_cell(self):
        if self.lstm_name == "lstm":
            lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_units, state_is_tuple=True)
        elif self.lstm_name == "gru":
            lstm_cell = tf.contrib.rnn.GRUCell(self.lstm_units)
        else:
            raise ValueError("LSTM name `{}` is illegal, choose from ['lstm', 'gru'].".format(self.lstm_name))

        if self.keep_prob < 1.0:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=1.0 - self.dropout_ratio_placeholder)
        return lstm_cell

    def _two_dense_layers(self, flat):
        final_dense_layer_1 = tf.layers.dense(flat, self.dense_units)
        final_dense_layer_1 = tf.layers.dropout(
            inputs=final_dense_layer_1,
            rate=self.dropout_ratio_placeholder,
            training=self.training_flag
        )
        final_dense_layer_2 = tf.layers.dense(final_dense_layer_1, self.dense_units)
        final_dense_layer_2 = tf.layers.dropout(
            inputs=final_dense_layer_2,
            rate=self.dropout_ratio_placeholder,
            training=self.training_flag
        )
        return final_dense_layer_2

    def _predict_loss(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.labels))
        if self.add_l2:
            l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
            self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
                self.loss + l2 * self.weight_decay
            )
        else:
            self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        logger.info("labels: {}, pred: {}".format(self.labels.shape, self.pred.shape))
        self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def _lstm_last_output(self, inputs):
        cell = tf.contrib.rnn.MultiRNNCell(
            [self._create_one_cell() for _ in range(self.lstm_num_layers)],
            state_is_tuple=True
        ) if self.lstm_num_layers > 1 else self._create_one_cell()

        # run dynamic RNN
        outputs, states = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, scope="dynamic_rnn")

        # before transpose, val.get_shape() = (batch_size, time_steps, num_units)
        # after transpose, val.get_shape() = (time_steps, batch_size, num_units)
        outputs = tf.transpose(outputs, [1, 0, 2])

        last_output = tf.gather(outputs, int(outputs.get_shape()[0]) - 1, name="lstm_last_state")
        logger.info("last_output shape: {}".format(last_output.shape))
        return last_output

    def build_graph_pure_lstm(self):
        self.inputs = tf.placeholder(tf.float32, [None, self.time_window, 1], name="inputs")
        self.labels = tf.placeholder(tf.int64, [None, self.num_classes], name="labels")
        self.training_flag = tf.placeholder(tf.bool)
        self.dropout_ratio_placeholder = tf.placeholder(tf.float32, name="dropout_ratio")
        logger.info("inputs: {}, labels: {}".format(self.inputs.shape, self.labels.shape))

        last_output = self._lstm_last_output(self.inputs)
        self.pred = tf.layers.dense(last_output, self.num_classes)
        self._predict_loss()

    def build_graph_vgg(self):
        self.inputs = tf.placeholder(
            tf.float32, [None, self.time_window, self.num_channels, self.num_wavelet_channels], name="inputs"
        )
        self.labels = tf.placeholder(tf.int64, [None, self.num_classes], name="labels")
        self.training_flag = tf.placeholder(tf.bool)
        self.dropout_ratio_placeholder = tf.placeholder(tf.float32, name="dropout_ratio")
        logger.info("inputs: {}, labels: {}".format(self.inputs.shape, self.labels.shape))

        # split inputs
        mag_inputs = tf.expand_dims(self.inputs[:, :, :, 0], -1)
        logger.info("mag_inputs: {}".format(mag_inputs.shape))

        flat = VGG(
            training=self.training_flag,
            keep_prob=1.0 - self.dropout_ratio_placeholder,
            num_layers=self.vgg_num_layers,
            for_attention=False,
            initial_filters=self.vgg_initial_filters
        ).build_vgg(mag_inputs)

        if self.two_dense_layers:
            final_dense_layer_2 = self._two_dense_layers(flat)
            self.pred = tf.layers.dense(final_dense_layer_2, self.num_classes)
        else:
            self.pred = tf.layers.dense(flat, self.num_classes)
        self._predict_loss()

    @staticmethod
    def _project_features(features, d_dimension, l_dimension):
        with tf.variable_scope('project_features'):
            w = tf.get_variable('w', [d_dimension, d_dimension], initializer=tf.contrib.layers.xavier_initializer())
            features_flat = tf.reshape(features, [-1, d_dimension])
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.reshape(features_proj, [-1, l_dimension, d_dimension])
            return features_proj

    def _attention_layer(self, original_features, h, h_dimension, l_dimension, d_dimension,
                         att_dimension="temp", reuse=False):
        with tf.variable_scope('attention_layer' + att_dimension, reuse=reuse):
            w = tf.get_variable('w', [h_dimension, d_dimension], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', [d_dimension], initializer=tf.constant_initializer(0.0))
            w_att = tf.get_variable('w_att', [d_dimension, 1], initializer=tf.contrib.layers.xavier_initializer())

            h_att = tf.nn.relu(self._project_features(original_features, d_dimension, l_dimension)
                               + tf.expand_dims(tf.matmul(h, w), 1) + b)    # (N, L, D)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, d_dimension]), w_att), [-1, l_dimension])   # (N, L)
            alpha = tf.nn.softmax(out_att)
            # logger.info("original_features: {}, alpha: {}".format(original_features.shape, alpha.shape))
            context = tf.reduce_sum(original_features * tf.expand_dims(alpha, 2), 1, name='context')  # (N, D)
            # logger.info("context: {}, alpha: {}".format(context.shape, alpha.shape))
            return context, alpha

    def build_graph_attention_vgg_lstm(self):
        self.inputs = tf.placeholder(
            tf.float32, [None, self.time_window, self.num_channels, self.num_wavelet_channels], name="inputs"
        )
        self.pure_lstm_inputs = tf.placeholder(
            tf.float32, [None, self.time_window, 1], name="pure_lstm_inputs"
        )
        self.labels = tf.placeholder(tf.float32, [None, self.num_classes], name="labels")
        self.training_flag = tf.placeholder(tf.bool)
        self.dropout_ratio_placeholder = tf.placeholder(tf.float32, name="dropout_ratio")

        vgg_flat = VGG(
            training=self.training_flag,
            keep_prob=1.0 - self.dropout_ratio_placeholder,
            num_layers=self.vgg_num_layers,
            for_attention=True,
            initial_filters=self.vgg_initial_filters
        ).build_vgg(self.inputs)
        logger.info("flat shape: {}".format(vgg_flat.shape))
        vgg_flat = tf.reshape(vgg_flat, [-1, int(vgg_flat.shape[1]) * int(vgg_flat.shape[2]), int(vgg_flat.shape[3])])

        last_output = self._lstm_last_output(self.pure_lstm_inputs)

        # add attention mechanism
        context, alpha = self._attention_layer(
            original_features=vgg_flat,
            h=last_output,
            h_dimension=(last_output.shape[1]),
            l_dimension=int(vgg_flat.shape[1]),
            d_dimension=int(vgg_flat.shape[2]),
            att_dimension="temp"
        )
        logger.info("context: {}, alpha: {}".format(context.shape, alpha.shape))

        # visualize attention weights
        if self.export_attention_weights:
            self.alpha_weights = alpha

        features_fusion = self._two_dense_layers(tf.concat([context, last_output]))
        self.pred = tf.layers.dense(features_fusion, self.num_classes)
        self._predict_loss()

    def build_graph_ensemble(self):
        self.inputs = tf.placeholder(
            tf.float32, [None, self.time_window, self.num_channels, self.num_wavelet_channels], name="inputs"
        )
        self.pure_lstm_inputs = tf.placeholder(
            tf.float32, [None, self.time_window, 1], name="pure_lstm_inputs"
        )
        self.labels = tf.placeholder(tf.float32, [None, self.num_classes], name="labels")
        self.training_flag = tf.placeholder(tf.bool)
        self.dropout_ratio_placeholder = tf.placeholder(tf.float32, name="dropout_ratio")

        vgg_flat = VGG(
            training=self.training_flag,
            keep_prob=1.0 - self.dropout_ratio_placeholder,
            num_layers=self.vgg_num_layers,
            for_attention=True,
            initial_filters=self.vgg_initial_filters
        ).build_vgg(self.inputs)
        logger.info("flat shape: {}".format(vgg_flat.shape))
        vgg_flat = tf.reshape(vgg_flat, [-1, int(vgg_flat.shape[1]) * int(vgg_flat.shape[2]), int(vgg_flat.shape[3])])

        last_output = self._lstm_last_output(self.pure_lstm_inputs)

        # flatten and dropout
        dense_cnn = self._two_dense_layers(vgg_flat)
        pred_cnn = tf.layers.dense(dense_cnn, self.num_classes)
        dense_lstm = tf.layers.dense(last_output, self.num_classes)
        self.pred = (1.0 - self.ensemble_lstm) * pred_cnn + self.ensemble_lstm * dense_lstm
        self._predict_loss()

    @staticmethod
    def _calculate_softmax(probs):
        e_x = np.exp(probs - np.max(probs))
        return e_x / e_x.sum()

    def _calculate_loss(self, data_x, data_y, data_x_lstm=None):
        num_iter = int(math.ceil(len(data_x) / self.batch_size))
        predict_y = np.zeros([len(data_y), self.num_classes])
        step = 0
        while step < num_iter:
            if data_x_lstm is None:
                feed_dict = {
                    self.inputs: data_x[step * self.batch_size:(step+1) * self.batch_size],
                    self.training_flag: False,
                    self.dropout_ratio_placeholder: 0.0
                }
            else:
                feed_dict = {
                    self.inputs: data_x[step * self.batch_size:(step+1) * self.batch_size],
                    self.pure_lstm_inputs: data_x_lstm[step * self.batch_size:(step+1) * self.batch_size],
                    self.training_flag: False,
                    self.dropout_ratio_placeholder: 0.0
                }
            predict_y_i = self.sess.run([self.pred], feed_dict)
            count = 0
            while count < self.batch_size and step * self.batch_size + count < len(data_x):
                predict_y_i_max = np.array(predict_y_i[0][count]).argmax()
                predict_y_i_count = np.zeros([self.num_classes], dtype=int)
                np.put(predict_y_i_count, predict_y_i_max, 1)
                predict_y[step * self.batch_size + count] = predict_y_i_count
                count += 1
            step += 1

        loss = log_loss(data_y, predict_y)
        accuracy = accuracy_score(data_y, predict_y)
        # auc = roc_auc_score(data_y, predict_y)
        return loss, predict_y, accuracy

    def train(self, data):
        random.seed(time.time())
        saver = tf.train.Saver()
        save_dir = os.path.join("../checkpoints", str(self.model_timestamp))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, "best_validation")
        best_val_loss = 1000 * 1000
        best_val_acc = -1

        init = tf.global_variables_initializer()
        self.sess.run(init)

        # val_data & test_data
        val_x_lstm = None
        test_x_lstm = None
        if self.model_structure in [1]:
            val_x, val_y = data.validation_lstm()
            test_x, test_y = data.testing_lstm()
        elif self.model_structure in [3, 4]:
            val_x, val_y = data.validation()
            val_x_lstm, val_y_lstm = data.validation_lstm()
            test_x, test_y = data.testing()
            test_x_lstm, test_y_lstm = data.testing_lstm()
        else:
            val_x, val_y = data.validation()
            test_x, test_y = data.testing()
        logger.info(
            "\nval: \nclass_0: {}, class_1: {}, "
            "\ntest: \nclass_0: {}, class_1: {}\n".format(
                np.sum(val_y[:, 0]), np.sum(val_y[:, 1]),
                np.sum(test_y[:, 0]), np.sum(test_y[:, 1])
            )
        )

        epoch = 0
        time_init = time.time()
        time_start = time_init

        train_size = data.train_size
        n_iters_per_epoch = int(np.ceil(float(train_size) / self.batch_size))

        while epoch < self.max_training_iters:
            step = 0
            time_step_start = time.time()
            train_loss_avg = 0.0
            train_acc_avg = 0.0
            train_loss_step = 0.0
            train_acc_step = 0.0

            while step < n_iters_per_epoch:
                if self.model_structure in [1]:
                    batch_x, batch_y = data.next_batch_lstm()
                    train_feed_dict = {
                        self.inputs: batch_x,
                        self.labels: batch_y,
                        self.training_flag: True,
                        self.dropout_ratio_placeholder: 1.0 - self.keep_prob
                    }
                elif self.model_structure in [3, 4]:
                    train_combined = data.next_batch_combined()
                    train_feed_dict = {
                        self.inputs: train_combined["batch_x_wavelet"],
                        self.pure_lstm_inputs: train_combined["batch_x_lstm"],
                        self.labels: train_combined["label"],
                        self.training_flag: True,
                        self.dropout_ratio_placeholder: 1.0 - self.keep_prob
                    }
                else:
                    batch_x, batch_y = data.next_batch()
                    train_feed_dict = {
                        self.inputs: batch_x,
                        self.labels: batch_y,
                        self.training_flag: True,
                        self.dropout_ratio_placeholder: 1.0 - self.keep_prob
                    }
                train_loss, train_optim, train_pred, train_acc = self.sess.run(
                    [self.loss, self.optim, self.pred, self.accuracy],
                    train_feed_dict
                )

                step += 1
                train_loss_avg += train_loss
                train_loss_step += train_loss
                train_acc_avg += train_acc
                train_acc_step += train_acc

                if step % self.display_step == 0:
                    logger.info("epoch: {}, step: {}, training loss: {:.8f}, train_acc: {:.4f}".format(
                        epoch, step, train_loss_step / self.display_step, train_acc_step / self.display_step
                    ))
                    val_loss, val_pred, val_acc = self._calculate_loss(val_x, val_y, val_x_lstm)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_val_acc = val_acc
                        saver.save(sess=self.sess, save_path=save_path)

                    logger.info(
                        "validation loss: {:.8f}, best validation loss: {:.8f}, "
                        "validation accuracy: {:.4f}, best validation accuracy: {:.4f}".format(
                            val_loss, best_val_loss,
                            val_acc, best_val_acc
                        )
                    )
                    logger.info("It takes {:.2f} seconds to run this step\n".format(time.time() - time_step_start))
                    time_step_start = time.time()
                    train_loss_step = 0.0
                    train_acc_step = 0.0

                # if not self.decay_dense_net and int(step * self.batch_size / 128) % 10000 == 0:
                #     self.learning_rate *= 0.1

            # evaluate each epoch
            val_loss, val_pred, val_acc = self._calculate_loss(val_x, val_y, val_x_lstm)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                saver.save(sess=self.sess, save_path=save_path)

            epoch += 1
            logger.info("epoch: {}, mini-batch training loss: {:.8f}, train_acc: {:.4f}".format(
                epoch,
                train_loss_avg / step,
                train_acc_avg / step
            ))
            logger.info(
                "validation loss: {:.8f}, best validation loss: {:.8f}, "
                "validation accuracy: {:.4f}, best validation accuracy: {:.4f}".format(
                    val_loss, best_val_loss,
                    val_acc, best_val_acc
                )
            )
            logger.info("It takes {:.2f} seconds to run this epoch.\n".format(time.time() - time_start))
            time_start = time.time()

            # reduce learning_rate
            if self.decay_dense_net:
                if epoch % int(0.5 * self.max_training_iters) == 0 or epoch % int(0.75 * self.max_training_iters) == 0:
                    self.learning_rate = self.learning_rate / 10.0

        logger.info("Optimization finished, best validation loss: {:.8f}, best validation accuracy: {:.4f}".format(
            best_val_loss, best_val_acc
        ))
        if self.restore_to_test:
            saver.restore(self.sess, save_path)

        # test_loss, test_optim, test_pred = self.sess.run([self.loss, self.optim, self.pred], test_feed_dict)
        test_time_start = time.time()
        test_loss, test_pred, test_acc = self._calculate_loss(test_x, test_y, test_x_lstm)
        logger.info("It takes {:.2f} seconds to test".format(time.time() - test_time_start))
        logger.info("test loss: {:8f}, test accuracy: {:.4f}".format(test_loss, test_acc))
        logger.info("It takes {:.2f} seconds to run this step".format(time.time() - time_init))

        return test_pred, test_y, best_val_loss, best_val_acc, test_loss, test_acc


class WANN(object):
    """Combine LSTM and CNN built on wavelet transform to predict future values."""
    def __init__(self, sess, ahead_step, batch_size, learning_rate,
                 keep_prob, time_window, num_channels, num_classes, max_training_iters=50,
                 display_step=100, model_structure=1, lstm_units=10, lstm_num_layers=1,
                 two_dense_layers=False, decay_dense_net=False, restore_to_test=True, vgg_num_layers=19,
                 dense_units=1024, vgg_initial_filters=64, training_slice=0, num_wavelet_channels=2,
                 add_l2=False, weight_decay=0.0001, lstm_name="lstm", ensemble_lstm=0.5,
                 export_attention_weights=False, model_timestamp=int(time.time() * 1000)):
        self.sess = sess
        self.ahead_step = ahead_step
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.time_window = time_window
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.max_training_iters = max_training_iters
        self.display_step = display_step
        self.model_structure = model_structure
        self.lstm_units = lstm_units
        self.lstm_num_layers = lstm_num_layers
        self.two_dense_layers = two_dense_layers
        self.decay_dense_net = decay_dense_net
        self.restore_to_test = restore_to_test
        self.vgg_num_layers = vgg_num_layers
        self.dense_units = dense_units
        self.vgg_initial_filters = vgg_initial_filters
        self.training_slice = training_slice
        self.num_wavelet_channels = num_wavelet_channels
        self.add_l2 = add_l2
        self.weight_decay = weight_decay
        self.lstm_name = lstm_name
        self.ensemble_lstm = ensemble_lstm
        self.export_attention_weights = export_attention_weights
        self.model_timestamp = model_timestamp

        self.inputs = None
        self.pure_lstm_inputs = None
        self.labels = None
        self.pred = None
        self.loss = None
        self.optim = None
        self.accuracy = None
        self.merged_summary_op = None
        self.training_flag = True
        self.dropout_ratio_placeholder = None
        self.alpha_weights = None

        if self.model_structure == 1:
            self.build_graph_pure_lstm()
        elif self.model_structure == 2:
            self.build_graph_vgg()
        elif self.model_structure == 3:
            self.build_graph_attention_vgg_lstm()
        elif self.model_structure == 4:
            self.build_graph_ensemble()
        elif self.model_structure == 5:
            self.build_graph_mlp()
        else:
            logger.info("Please specify the right model_structure number (choose from 1, 2, 3, 4)")
            exit(1)

    def _create_one_cell(self):
        if self.lstm_name == "lstm":
            lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_units, state_is_tuple=True)
        elif self.lstm_name == "gru":
            lstm_cell = tf.contrib.rnn.GRUCell(self.lstm_units)
        else:
            raise ValueError("LSTM name `{}` is illegal, choose from ['lstm', 'gru'].".format(self.lstm_name))

        if self.keep_prob < 1.0:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=1.0 - self.dropout_ratio_placeholder)
        return lstm_cell

    def _two_dense_layers(self, flat):
        final_dense_layer_1 = tf.layers.dense(flat, self.dense_units)
        final_dense_layer_1 = tf.layers.dropout(
            inputs=final_dense_layer_1,
            rate=self.dropout_ratio_placeholder,
            training=self.training_flag
        )
        final_dense_layer_2 = tf.layers.dense(final_dense_layer_1, self.dense_units)
        final_dense_layer_2 = tf.layers.dropout(
            inputs=final_dense_layer_2,
            rate=self.dropout_ratio_placeholder,
            training=self.training_flag
        )
        return final_dense_layer_2

    def _predict_loss(self):
        self.loss = tf.reduce_mean(tf.square(self.pred - self.labels))
        if self.add_l2:
            l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
            self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
                self.loss + l2 * self.weight_decay
            )
        else:
            self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def _lstm_last_output(self, inputs):
        cell = tf.contrib.rnn.MultiRNNCell(
            [self._create_one_cell() for _ in range(self.lstm_num_layers)],
            state_is_tuple=True
        ) if self.lstm_num_layers > 1 else self._create_one_cell()

        # run dynamic RNN
        outputs, states = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, scope="dynamic_rnn")

        # before transpose, val.get_shape() = (batch_size, time_steps, num_units)
        # after transpose, val.get_shape() = (time_steps, batch_size, num_units)
        outputs = tf.transpose(outputs, [1, 0, 2])

        last_output = tf.gather(outputs, int(outputs.get_shape()[0]) - 1, name="lstm_last_state")
        logger.info("last_output shape: {}".format(last_output.shape))
        return last_output

    def build_graph_pure_lstm(self):
        self.inputs = tf.placeholder(tf.float32, [None, self.time_window, 1], name="inputs")
        self.labels = tf.placeholder(tf.float32, [None, self.num_classes], name="labels")
        self.training_flag = tf.placeholder(tf.bool)
        self.dropout_ratio_placeholder = tf.placeholder(tf.float32, name="dropout_ratio")
        logger.info("inputs: {}, labels: {}".format(self.inputs.shape, self.labels.shape))

        last_output = self._lstm_last_output(self.inputs)
        self.pred = tf.layers.dense(last_output, self.num_classes)
        self._predict_loss()

    @staticmethod
    def _mlp(inputs, num_layers, num_neurons):
        assert num_layers >= 1
        hidden_output = tf.layers.dense(inputs, num_neurons)
        for layer_i in range(0, num_layers - 1):
            hidden_output = tf.layers.dense(hidden_output, num_neurons)
        return hidden_output

    def build_graph_mlp(self):
        self.inputs = tf.placeholder(tf.float32, [None, self.time_window, 1], name="inputs")
        self.labels = tf.placeholder(tf.float32, [None, self.num_classes], name="labels")
        self.training_flag = tf.placeholder(tf.bool)
        self.dropout_ratio_placeholder = tf.placeholder(tf.float32, name="dropout_ratio")
        logger.info("inputs: {}, labels: {}".format(self.inputs.shape, self.labels.shape))

        inputs = tf.reshape(self.inputs, [-1, self.time_window])
        hidden_output = self._mlp(inputs, self.lstm_num_layers, self.dense_units)
        self.pred = tf.layers.dense(hidden_output, self.num_classes)
        logger.info("hidden_output: {}, self.pred: {}, self.labels: {}".format(
            hidden_output.shape, self.pred.shape, self.labels.shape
        ))
        self._predict_loss()

    def build_graph_vgg(self):
        self.inputs = tf.placeholder(
            tf.float32, [None, self.time_window, self.num_channels, self.num_wavelet_channels], name="inputs"
        )
        self.labels = tf.placeholder(tf.float32, [None, self.num_classes], name="labels")
        self.training_flag = tf.placeholder(tf.bool)
        self.dropout_ratio_placeholder = tf.placeholder(tf.float32, name="dropout_ratio")
        logger.info("inputs: {}, labels: {}".format(self.inputs.shape, self.labels.shape))

        # split inputs
        mag_inputs = tf.expand_dims(self.inputs[:, :, :, 0], -1)
        logger.info("mag_inputs: {}".format(mag_inputs.shape))

        flat = VGG(
            training=self.training_flag,
            keep_prob=1.0 - self.dropout_ratio_placeholder,
            num_layers=self.vgg_num_layers,
            for_attention=False,
            initial_filters=self.vgg_initial_filters
        ).build_vgg(mag_inputs)

        if self.two_dense_layers:
            final_dense_layer_2 = self._two_dense_layers(flat)
            self.pred = tf.layers.dense(final_dense_layer_2, self.num_classes)
        else:
            self.pred = tf.layers.dense(flat, self.num_classes)
        self._predict_loss()

    @staticmethod
    def _project_features(features, d_dimension, l_dimension):
        with tf.variable_scope('project_features'):
            w = tf.get_variable('w', [d_dimension, d_dimension], initializer=tf.contrib.layers.xavier_initializer())
            features_flat = tf.reshape(features, [-1, d_dimension])
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.reshape(features_proj, [-1, l_dimension, d_dimension])
            return features_proj

    def _attention_layer(self, original_features, h, h_dimension, l_dimension, d_dimension,
                         att_dimension="temp", reuse=False):
        with tf.variable_scope('attention_layer' + att_dimension, reuse=reuse):
            w = tf.get_variable('w', [h_dimension, d_dimension], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', [d_dimension], initializer=tf.constant_initializer(0.0))
            w_att = tf.get_variable('w_att', [d_dimension, 1], initializer=tf.contrib.layers.xavier_initializer())

            h_att = tf.nn.relu(self._project_features(original_features, d_dimension, l_dimension)
                               + tf.expand_dims(tf.matmul(h, w), 1) + b)    # (N, L, D)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, d_dimension]), w_att), [-1, l_dimension])   # (N, L)
            alpha = tf.nn.softmax(out_att)
            # logger.info("original_features: {}, alpha: {}".format(original_features.shape, alpha.shape))
            context = tf.reduce_sum(original_features * tf.expand_dims(alpha, 2), 1, name='context')  # (N, D)
            # logger.info("context: {}, alpha: {}".format(context.shape, alpha.shape))
            return context, alpha

    def build_graph_attention_vgg_lstm(self):
        self.inputs = tf.placeholder(
            tf.float32, [None, self.time_window, self.num_channels, self.num_wavelet_channels], name="inputs"
        )
        self.pure_lstm_inputs = tf.placeholder(
            tf.float32, [None, self.time_window, 1], name="pure_lstm_inputs"
        )
        self.labels = tf.placeholder(tf.float32, [None, self.num_classes], name="labels")
        self.training_flag = tf.placeholder(tf.bool)
        self.dropout_ratio_placeholder = tf.placeholder(tf.float32, name="dropout_ratio")

        vgg_flat = VGG(
            training=self.training_flag,
            keep_prob=1.0 - self.dropout_ratio_placeholder,
            num_layers=self.vgg_num_layers,
            for_attention=True,
            initial_filters=self.vgg_initial_filters
        ).build_vgg(self.inputs)
        logger.info("flat shape: {}".format(vgg_flat.shape))
        vgg_flat = tf.reshape(vgg_flat, [-1, int(vgg_flat.shape[1]) * int(vgg_flat.shape[2]), int(vgg_flat.shape[3])])

        last_output = self._lstm_last_output(self.pure_lstm_inputs)

        # add attention mechanism
        context, alpha = self._attention_layer(
            original_features=vgg_flat,
            h=last_output,
            h_dimension=(last_output.shape[1]),
            l_dimension=int(vgg_flat.shape[1]),
            d_dimension=int(vgg_flat.shape[2]),
            att_dimension="temp"
        )
        logger.info("context: {}, alpha: {}".format(context.shape, alpha.shape))

        # visualize attention weights
        if self.export_attention_weights:
            self.alpha_weights = alpha

        self.pred = self._two_dense_layers(tf.concat([context, last_output]))
        self._predict_loss()

    def build_graph_ensemble(self):
        self.inputs = tf.placeholder(
            tf.float32, [None, self.time_window, self.num_channels, self.num_wavelet_channels], name="inputs"
        )
        self.pure_lstm_inputs = tf.placeholder(
            tf.float32, [None, self.time_window, 1], name="pure_lstm_inputs"
        )
        self.labels = tf.placeholder(tf.float32, [None, self.num_classes], name="labels")
        self.training_flag = tf.placeholder(tf.bool)
        self.dropout_ratio_placeholder = tf.placeholder(tf.float32, name="dropout_ratio")

        vgg_flat = VGG(
            training=self.training_flag,
            keep_prob=1.0 - self.dropout_ratio_placeholder,
            num_layers=self.vgg_num_layers,
            for_attention=True,
            initial_filters=self.vgg_initial_filters
        ).build_vgg(self.inputs)
        logger.info("flat shape: {}".format(vgg_flat.shape))
        vgg_flat = tf.reshape(vgg_flat, [-1, int(vgg_flat.shape[1]) * int(vgg_flat.shape[2]) * int(vgg_flat.shape[3])])

        last_output = self._lstm_last_output(self.pure_lstm_inputs)

        # flatten and dropout
        dense_cnn = self._two_dense_layers(vgg_flat)
        pred_cnn = tf.layers.dense(dense_cnn, self.num_classes)
        dense_lstm = tf.layers.dense(last_output, self.num_classes)
        self.pred = (1.0 - self.ensemble_lstm) * pred_cnn + self.ensemble_lstm * dense_lstm
        self._predict_loss()

    def _calculate_loss(self, data_x, data_y, data_x_lstm=None):
        num_iter = int(math.ceil(len(data_x) / self.batch_size))
        predict_y = np.zeros([len(data_y), self.num_classes])
        step = 0
        while step < num_iter:
            if data_x_lstm is None:
                feed_dict = {
                    self.inputs: data_x[step * self.batch_size:(step+1) * self.batch_size],
                    self.training_flag: False,
                    self.dropout_ratio_placeholder: 0.0
                }
            else:
                feed_dict = {
                    self.inputs: data_x[step * self.batch_size:(step+1) * self.batch_size],
                    self.pure_lstm_inputs: data_x_lstm[step * self.batch_size:(step+1) * self.batch_size],
                    self.training_flag: False,
                    self.dropout_ratio_placeholder: 0.0
                }
            predict_y_i = self.sess.run([self.pred], feed_dict)
            count = 0
            while count < self.batch_size and step * self.batch_size + count < len(data_x):
                predict_y[step * self.batch_size + count] = predict_y_i[0][count]
                count += 1
            step += 1

        loss = mean_squared_error(data_y, predict_y)
        return loss, predict_y

    def train(self, data):
        random.seed(time.time())
        saver = tf.train.Saver()
        save_dir = os.path.join("../checkpoints", str(self.model_timestamp))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, "best_validation")
        best_val_loss = 1000 * 1000

        init = tf.global_variables_initializer()
        self.sess.run(init)

        # val_data & test_data
        val_x_lstm = None
        test_x_lstm = None
        if self.model_structure in [1, 5]:
            val_x, val_y = data.validation_lstm()
            test_x, test_y = data.testing_lstm()
        elif self.model_structure in [3, 4]:
            val_x, val_y = data.validation()
            val_x_lstm, val_y_lstm = data.validation_lstm()
            test_x, test_y = data.testing()
            test_x_lstm, test_y_lstm = data.testing_lstm()
        else:
            val_x, val_y = data.validation()
            test_x, test_y = data.testing()

        # for debugging
        logger.info("val_x: {}, val_y: {}".format(val_x.shape, val_y.shape))

        epoch = 0
        time_init = time.time()
        time_start = time_init

        train_size = data.train_size
        n_iters_per_epoch = int(np.ceil(float(train_size) / self.batch_size))

        while epoch < self.max_training_iters:
            step = 0
            time_step_start = time.time()
            train_loss_avg = 0.0
            train_loss_step = 0.0

            while step < n_iters_per_epoch:
                if self.model_structure in [1, 5]:
                    batch_x, batch_y = data.next_batch_lstm()
                    train_feed_dict = {
                        self.inputs: batch_x,
                        self.labels: batch_y,
                        self.training_flag: True,
                        self.dropout_ratio_placeholder: 1.0 - self.keep_prob
                    }
                elif self.model_structure in [3, 4]:
                    train_combined = data.next_batch_combined()
                    train_feed_dict = {
                        self.inputs: train_combined["batch_x_wavelet"],
                        self.pure_lstm_inputs: train_combined["batch_x_lstm"],
                        self.labels: train_combined["label"],
                        self.training_flag: True,
                        self.dropout_ratio_placeholder: 1.0 - self.keep_prob
                    }
                else:
                    batch_x, batch_y = data.next_batch()
                    train_feed_dict = {
                        self.inputs: batch_x,
                        self.labels: batch_y,
                        self.training_flag: True,
                        self.dropout_ratio_placeholder: 1.0 - self.keep_prob
                    }
                train_loss, train_optim, train_pred = self.sess.run(
                    [self.loss, self.optim, self.pred],
                    train_feed_dict
                )

                step += 1
                train_loss_avg += train_loss
                train_loss_step += train_loss

                if step % self.display_step == 0:
                    logger.info("epoch: {}, step: {}, training loss: {:.8f}".format(
                        epoch, step, train_loss_step / self.display_step
                    ))
                    val_loss, val_pred = self._calculate_loss(val_x, val_y, val_x_lstm)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        saver.save(sess=self.sess, save_path=save_path)

                    logger.info(
                        "validation loss: {:.8f}, best validation loss: {:.8f}".format(
                            val_loss, best_val_loss,
                        )
                    )
                    logger.info("It takes {:.2f} seconds to run this step\n".format(time.time() - time_step_start))
                    time_step_start = time.time()
                    train_loss_step = 0.0

                # if not self.decay_dense_net and int(step * self.batch_size / 128) % 10000 == 0:
                #     self.learning_rate *= 0.1

            # evaluate each epoch
            val_loss, val_pred = self._calculate_loss(val_x, val_y, val_x_lstm)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                saver.save(sess=self.sess, save_path=save_path)

            epoch += 1
            logger.info("epoch: {}, mini-batch training loss: {:.8f}".format(
                epoch,
                train_loss_avg / step
            ))
            logger.info(
                "validation loss: {:.8f}, best validation loss: {:.8f}".format(
                    val_loss, best_val_loss
                )
            )
            logger.info("It takes {:.2f} seconds to run this epoch.\n".format(time.time() - time_start))
            time_start = time.time()

            # reduce learning_rate
            if self.decay_dense_net:
                if epoch % int(0.5 * self.max_training_iters) == 0 or epoch % int(0.75 * self.max_training_iters) == 0:
                    self.learning_rate = self.learning_rate / 10.0

        logger.info("Optimization finished, best validation loss: {:.8f}".format(
            best_val_loss
        ))
        if self.restore_to_test:
            saver.restore(self.sess, save_path)

        # test_loss, test_optim, test_pred = self.sess.run([self.loss, self.optim, self.pred], test_feed_dict)
        test_time_start = time.time()
        test_loss, test_pred = self._calculate_loss(test_x, test_y, test_x_lstm)
        logger.info("It takes {:.2f} seconds to test".format(time.time() - test_time_start))
        logger.info("It takes {:.2f} seconds to run this step".format(time.time() - time_init))

        return test_pred, test_y, best_val_loss, test_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_structure", help="choose which cnn structure to train", required=False, default=1, type=int)
    parser.add_argument("--vgg_num_layers", help="number of vgg layers", required=False, default=19, type=int)
    args = parser.parse_args()

    tf.reset_default_graph()
    with tf.Session() as sess:
        cnn_model = WaveletAttentionNetwork(
            sess,
            ahead_step=1,
            batch_size=50,
            learning_rate=0.0001,
            keep_prob=0.9,
            time_window=32,
            num_channels=32,
            num_classes=1,
            max_training_iters=5,
            model_structure=args.model_structure,
            lstm_units=10,
            two_dense_layers=False,
            vgg_num_layers=args.vgg_num_layers
        )
