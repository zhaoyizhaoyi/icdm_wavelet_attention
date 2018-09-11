import numpy as np
import pandas as pd
import os
import logging
import random
import pywt
import time
from multiprocessing import Pool
import sys
import tensorflow as tf
import datetime
import argparse
sys.path.append("../")
from core import WaveletAttentionNetwork

# configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockData(object):
    """
    Read stock price data from data directory and reshape them to what we need.
    """
    def __init__(self, data_dir, period_len, target_column, ahead_step, train_percentage, val_percentage, time_window,
                 num_classes, num_frequencies, batch_size, model_structure=1, num_slave=20, wavelet_function="cmor",
                 training_slice=0, num_wavelet_channels=2, use_amplitude=0, standardize_method=""):
        self.data_dir = data_dir
        self.period_len = period_len
        self.target_column = target_column
        self.ahead_step = ahead_step
        self.train_percentage = train_percentage
        self.val_percentage = val_percentage
        self.time_window = time_window
        self.num_classes = num_classes
        self.num_frequencies = num_frequencies
        self.batch_size = batch_size
        self.model_structure = model_structure
        self.num_slave = num_slave
        self.wavelet_function = wavelet_function
        self.training_slice = training_slice
        self.num_wavelet_channels = num_wavelet_channels
        self.use_amplitude = use_amplitude
        self.standardize_method = standardize_method

        self.data = self._load_data()
        self.normalized_data = None

        self.train_index = None
        self.val_index = None
        self.test_index = None

        self.min_data = None
        self.max_data = None

        self.train_size = 0
        self.val_size = 0
        self.test_size = 0

        self.train_j_item_list = []
        self._split_data()

        # check to see whether wavelet transform has been computed already
        self.wavelet_x = None
        if self.model_structure not in [1]:
            self.save_load_wavelet()

    @staticmethod
    def _load_data():
        data = np.load('../data/stock/data.npy')
        return data

    def _generate_tuple_list(self, index):
        j_item_tuple_list = []
        for j in range(len(self.data)):
            for item in index:
                j_item_tuple_list.append((j, item))
        return j_item_tuple_list

    def _split_data(self):
        # maybe we should write it like this
        train_pos = int(self.train_percentage * self.data.shape[1])
        val_pos = int(self.val_percentage * self.data.shape[1]) + train_pos

        self.train_index = list(range(0, train_pos - self.time_window - self.ahead_step + 1))
        self.val_index = list(range(train_pos - self.time_window - self.ahead_step + 1, val_pos - self.time_window - self.ahead_step + 1))
        self.test_index = list(range(val_pos - self.time_window - self.ahead_step + 1, self.data.shape[1] - self.time_window - self.ahead_step + 1))

        # data normalization
        self.max_data = np.max(self.data, axis=1)
        self.min_data = np.min(self.data, axis=1)
        self.max_data = np.reshape(self.max_data, (self.max_data.shape[0], 1))
        self.min_data = np.reshape(self.min_data, (self.min_data.shape[0], 1))
        self.normalized_data = (self.data - self.min_data) / (self.max_data - self.min_data)

        self.train_size = len(self.train_index) * len(self.data)
        self.val_size = len(self.val_index)
        self.test_size = len(self.test_index)
        self.train_j_item_list = self._generate_tuple_list(self.train_index)

        logger.info(
            "\nSplit data finished!\n"
            "train: {}, val: {}, test: {}, \ntrain_tuple: {}, "
            "train_tuple[0]: {}".format(
                self.train_size, self.val_size, self.test_size,
                len(self.train_j_item_list), self.train_j_item_list[0]
            )
        )

    def _generate_time_frequency(self, time_series):
        wavelet_transformed, freqs = pywt.cwt(time_series, range(1, self.num_frequencies + 1), self.wavelet_function)
        return wavelet_transformed

    def _generate_batch_single(self, index):
        index_size = len(index)

        # the shape of batch_x, label
        batch_x = np.zeros([index_size * len(self.data), self.time_window, self.num_frequencies])
        label = np.zeros([index_size * len(self.data), self.num_classes])

        temp = 0
        for j in range(len(self.data)):
            for item in index:
                batch_x[temp, :, :] = self._generate_time_frequency(self.normalized_data[j][item:item + self.time_window])
                if self.normalized_data[j][item + self.time_window + self.ahead_step - 1] >= self.normalized_data[j][item + self.time_window - 1]:
                    label[temp, 1] = 1.
                else:
                    label[temp, 0] = 1.
                temp += 1
        return batch_x, label

    def _generate_x_y_pair(self, j_item_tuple):
        """
        To facilitate the multi processing.
        """
        j = j_item_tuple[0]
        item = j_item_tuple[1]
        if self.wavelet_x is None:
            wavelet_transformed = self._generate_time_frequency(self.normalized_data[j][item:item + self.time_window])
            if self.num_wavelet_channels > 1:
                if self.use_amplitude == 1:
                    batch_x_i = np.stack((np.absolute(wavelet_transformed), np.arctan(wavelet_transformed.imag / wavelet_transformed.real)), axis=-1)
                else:
                    batch_x_i = np.stack((wavelet_transformed.real, wavelet_transformed.imag), axis=-1)
            else:
                batch_x_i = wavelet_transformed
        else:
            batch_x_i = self.wavelet_x[j][item]

        if self.normalized_data[j][item + self.time_window + self.ahead_step - 1] >= self.normalized_data[j][item + self.time_window - 1]:
            label_i = np.array([0., 1.])
        else:
            label_i = np.array([1., 0.])
        batch_tuple_i = (batch_x_i, label_i, j, item)

        return batch_tuple_i

    def _pre_compute_save(self):
        index = np.array(range(0, self.data.shape[1] - self.time_window))
        index_size = len(index)

        # the shape of batch_x
        batch_x = np.zeros([len(self.data), index_size, self.time_window, self.num_frequencies, self.num_wavelet_channels])

        j_item_tuple_list = []
        for j in range(len(self.data)):
            for item in index:
                j_item_tuple_list.append((j, item))

        p = Pool(self.num_slave)
        outputs = p.map(self._generate_x_y_pair, j_item_tuple_list)
        p.close()
        p.join()
        for temp in range(len(outputs)):
            batch_x[outputs[temp][2], outputs[temp][3], :, :] = outputs[temp][0]
        return batch_x

    def save_load_wavelet(self):
        wavelet_dir = "./../data/wavelet_pre_computed/stock/"
        if not os.path.exists(wavelet_dir):
            os.makedirs(wavelet_dir)

        if self.standardize_method == "":
            wavelet_filename = "stock_wavelet_" + self.wavelet_function + "_" + str(self.time_window) + "_" + str(self.num_frequencies) + "_" + str(self.use_amplitude) + ".npy"
        else:
            wavelet_filename = "stock_wavelet_" + self.wavelet_function + "_" + str(self.time_window) + "_" + str(
                self.num_frequencies) + "_" + str(self.use_amplitude) + "_" + self.standardize_method + ".npy"

        wavelet_filepath = os.path.join(wavelet_dir, wavelet_filename)
        if not os.path.exists(wavelet_filepath):
            self.wavelet_x = self._pre_compute_save()
            logger.info("saving wavelet_x to file {}...".format(wavelet_filepath))
            np.save(wavelet_filepath, self.wavelet_x)
        else:
            logger.info("loading wavelet_x from file {}...".format(wavelet_filepath))
            self.wavelet_x = np.load(wavelet_filepath)

    def _generate_batch(self, index):
        index_size = len(index)

        # the shape of batch_x, label
        batch_x = np.zeros([index_size * len(self.data), self.time_window, self.num_frequencies, self.num_wavelet_channels])
        label = np.zeros([index_size * len(self.data), self.num_classes])

        j_item_tuple_list = []
        for j in range(len(self.data)):
            for item in index:
                j_item_tuple_list.append((j, item))

        p = Pool(self.num_slave)
        outputs = p.map(self._generate_x_y_pair, j_item_tuple_list)
        p.close()
        p.join()
        for temp in range(len(outputs)):
            batch_x[temp, :, :] = outputs[temp][0]
            label[temp, :] = outputs[temp][1]
        return batch_x, label

    def _generate_batch_from_pre_compute(self, tuple_list):
        index_size = len(tuple_list)

        # the shape of batch_x, label
        batch_x = np.zeros([index_size, self.time_window, self.num_frequencies, self.num_wavelet_channels])
        label = np.zeros([index_size, self.num_classes])

        temp = 0
        for tuple_i in tuple_list:
            j = tuple_i[0]
            item = tuple_i[1]
            batch_x[temp, :, :] = self.wavelet_x[j][item]
            if self.normalized_data[j][item + self.time_window + self.ahead_step - 1] >= self.normalized_data[j][item + self.time_window - 1]:
                label[temp, 1] = 1.
            else:
                label[temp, 0] = 1.
            temp += 1

        return batch_x, label

    def next_batch(self):
        # generate a random index from the range [0, len(self.train_x) - self.time_window]
        index = random.sample(self.train_index, self.batch_size)
        # index = range(0, len(self.train_x[1]))
        index = np.array(index)
        # index = np.array(range(0, len(self.train_x[1])))
        if self.wavelet_x is None:
            batch_x, label = self._generate_batch(index)
        else:
            tuple_list = random.sample(self.train_j_item_list, self.batch_size)
            batch_x, label = self._generate_batch_from_pre_compute(tuple_list)
        return batch_x, label

    def validation(self):
        index = np.array(self.val_index)
        # logger.info("index max: {}, min: {}".format(np.max(index), np.min(index)))
        # logger.info("max y pos: {}".format(np.max(index) + self.time_window + self.ahead_step - 1))
        if self.wavelet_x is None:
            batch_x, label = self._generate_batch(index)
        else:
            val_tuple_list = self._generate_tuple_list(self.val_index)
            batch_x, label = self._generate_batch_from_pre_compute(val_tuple_list)
        return batch_x, label

    def testing(self):
        index = np.array(self.test_index)
        # logger.info("index max: {}, min: {}".format(np.max(index), np.min(index)))
        # logger.info("max y pos: {}".format(np.max(index) + self.time_window + self.ahead_step - 1))
        if self.wavelet_x is None:
            batch_x, label = self._generate_batch(index)
        else:
            test_tuple_list = self._generate_tuple_list(self.test_index)
            batch_x, label = self._generate_batch_from_pre_compute(test_tuple_list)
        return batch_x, label

    def _generate_batch_lstm(self, tuple_list):
        index_size = len(tuple_list)

        # the shape of batch_x, label
        batch_x = np.zeros([index_size, self.time_window, 1])
        label = np.zeros([index_size, self.num_classes])

        temp = 0
        for tuple_i in tuple_list:
            j = tuple_i[0]
            item = tuple_i[1]
            batch_x[temp, :, :] = np.reshape(self.normalized_data[j][item:item + self.time_window], (-1, 1))
            if self.normalized_data[j][item + self.time_window + self.ahead_step - 1] >= self.normalized_data[j][item + self.time_window - 1]:
                label[temp, 1] = 1.
            else:
                label[temp, 0] = 1.
            temp += 1
        return batch_x, label

    def next_batch_lstm(self):
        # index = random.sample(self.train_index, self.batch_size)
        tuple_list = random.sample(self.train_j_item_list, self.batch_size)
        # index = np.array(index)
        batch_x, label = self._generate_batch_lstm(tuple_list)
        return batch_x, label

    def validation_lstm(self):
        # index = np.array(self.val_index)
        val_tuple_list = self._generate_tuple_list(self.val_index)
        batch_x, label = self._generate_batch_lstm(val_tuple_list)
        return batch_x, label

    def testing_lstm(self):
        # index = np.array(self.test_index)
        test_tuple_list = self._generate_tuple_list(self.test_index)
        batch_x, label = self._generate_batch_lstm(test_tuple_list)
        return batch_x, label

    def next_batch_combined(self):
        """We make this next_batch_combined single, because it involves a random process.
        Therefore, we cannot make the batch_x and label the same unless we provide the same index
        to generate_batch method.
        """
        index = random.sample(self.train_index, self.batch_size)
        index = np.array(index)

        tuple_list = random.sample(self.train_j_item_list, self.batch_size)

        # for lstm
        batch_x_lstm, label_lstm = self._generate_batch_lstm(tuple_list)

        # for wavelet
        if self.wavelet_x is None:
            batch_x_wavelet, label_wavelet = self._generate_batch(index)
        else:
            batch_x_wavelet, label_wavelet = self._generate_batch_from_pre_compute(tuple_list)

        return {
            "batch_x_lstm": batch_x_lstm,
            "batch_x_wavelet": batch_x_wavelet,
            "label": label_lstm
        }


if __name__ == "__main__":
    today = datetime.date.today().strftime("%Y%m%d")
    logger.info("Running model on {}.".format(today))

    parser = argparse.ArgumentParser()
    parser.add_argument("--ahead_step", help="time step to predict", required=False, default=1, type=int)
    parser.add_argument("--time_window", help="time window to look back", required=False, default=5, type=int)
    parser.add_argument("--num_frequencies", help="number of Fourier frequencies to decompose the time series",
                        required=False, default=5, type=int)
    parser.add_argument("--batch_size", help="size of batch to train", required=False, default=64, type=int)
    parser.add_argument("--learning_rate", help="learning rate", required=False, default=0.001, type=float)
    parser.add_argument("--keep_prob", help="dropout rate", required=False, default=0.9, type=float)
    parser.add_argument("--ensemble_lstm", help="ensemble lstm weight", required=False, default=0.5, type=float)
    parser.add_argument("--max_training_iters", help="max training iterations", required=False, default=50,
                        type=int)
    parser.add_argument("--model_structure", help="choose which model structure to train", required=False, default=1,
                        type=int)
    parser.add_argument("--lstm_units", help="number of hidden states in lstm", required=False, default=10,
                        type=int)
    parser.add_argument("--lstm_num_layers", help="number of lstm layers", required=False, default=1, type=int)
    parser.add_argument("--wavelet_function", help="wavelet function", required=False, default="cmor", type=str)
    parser.add_argument("--use_amplitude", help="whether to use amplitude or not", required=False, default=0,
                        type=int)
    parser.add_argument("--gpu_fraction", help="how much fraction of gpu to use", required=False, default=0.5,
                        type=float)
    parser.add_argument("--decay_dense_net",
                        help="decay learning rate only at 50% and 75% percentage of max_training_iters",
                        required=False, default=False, type=lambda x: (str(x).lower() == "true"))
    parser.add_argument("--restore_to_test", help="whether test from restored model", required=False, default=True,
                        type=lambda x: (str(x).lower() == "true"))
    parser.add_argument("--two_dense_layers", help="whether to use one more layer, default false", required=False,
                        default=False, type=lambda x: (str(x).lower() == "true"))
    parser.add_argument("--dense_units", help="number of units in first fully connected dense layer",
                        required=False, default=1024, type=int)
    parser.add_argument("--vgg_num_layers", help="number of vgg layers", required=False, default=19, type=int)
    parser.add_argument("--vgg_kernel_size", help="size of vgg kernel", required=False, default=3, type=int)
    parser.add_argument("--vgg_initial_filters", help="number of initial filters in VGG", required=False,
                        default=64, type=int)
    parser.add_argument("--training_slice", help="desert training data before training_slice", required=False,
                        default=0, type=int)
    parser.add_argument("--num_wavelet_channels", help="number of wavelet channels", required=False, default=2,
                        type=int)
    parser.add_argument("--add_l2", help="whether add l2 regularization or not", required=False, default=False,
                        type=lambda x: (str(x).lower() == "true"))
    parser.add_argument("--weight_decay", help="weight decay of normalization", required=False, default=0.0001,
                        type=float)
    parser.add_argument("--gpu_device", help="which gpu to use", required=False, default="0", type=str)
    parser.add_argument("--display_step", help="compute validation loss every display_step steps", required=False,
                        default=100, type=int)
    parser.add_argument("--standardize_method", help="standardized method", required=False, default="", type=str)
    parser.add_argument("--notes", help="specify the model structure", required=False, default="", type=str)
    parser.add_argument("--lstm_name", help="which type of lstm, choose from [`lstm`, `gru`]", required=False,
                        default="lstm", type=str)
    parser.add_argument("--export_attention_weights", help="whether to export attention weights or not",
                        required=False, default=False, type=lambda x: (str(x).lower() == "true"))
    args = parser.parse_args()

    # set which gpu to use
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    time_start = time.time()

    if args.gpu_fraction < 1.0:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
        config = tf.ConfigProto(gpu_options=gpu_options)
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

    tf.reset_default_graph()
    with tf.Session(config=config) as sess:
        # import data
        stock_data = StockData(
            data_dir="./../data/price_long_50/",
            period_len=2518,
            target_column="Open",
            ahead_step=args.ahead_step,
            train_percentage=0.8,
            val_percentage=0.1,
            time_window=args.time_window,
            num_classes=2,
            num_frequencies=args.num_frequencies,
            batch_size=args.batch_size,
            model_structure=args.model_structure,
            wavelet_function=args.wavelet_function,
            training_slice=args.training_slice,
            num_wavelet_channels=args.num_wavelet_channels,
            use_amplitude=args.use_amplitude,
            standardize_method=args.standardize_method
        )

        # for looking checkpoints and log
        model_timestamp = int(time.time() * 1000)
        logger.info("Model timestamp: {}".format(str(model_timestamp)))

        cnn_model = WaveletAttentionNetwork(
            sess=sess,
            ahead_step=args.ahead_step,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            keep_prob=args.keep_prob,
            time_window=args.time_window,
            num_channels=args.num_frequencies,
            num_classes=2,
            max_training_iters=args.max_training_iters,
            display_step=args.display_step,
            model_structure=args.model_structure,
            lstm_units=args.lstm_units,
            lstm_num_layers=args.lstm_num_layers,
            two_dense_layers=args.two_dense_layers,
            decay_dense_net=args.decay_dense_net,
            restore_to_test=args.restore_to_test,
            vgg_num_layers=args.vgg_num_layers,
            dense_units=args.dense_units,
            vgg_initial_filters=args.vgg_initial_filters,
            training_slice=args.training_slice,
            num_wavelet_channels=args.num_wavelet_channels,
            add_l2=args.add_l2,
            weight_decay=args.weight_decay,
            lstm_name=args.lstm_name,
            ensemble_lstm=args.ensemble_lstm,
            export_attention_weights=args.export_attention_weights,
            model_timestamp=model_timestamp
        )

        test_pred, test_y, best_val_loss, best_val_acc, test_loss, test_acc = cnn_model.train(stock_data)

        # wavelet function
        notes = args.notes + "_wavelet_" + args.wavelet_function

        # specify results path
        result_dir = "../results/stock/"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_path = os.path.join(result_dir, 'stock_results_step_' + str(args.ahead_step) + "_new.csv")

        if not os.path.exists(result_path):
            with open(result_path, 'a') as file:
                file.write("ahead_step,model_structure,time_window,num_frequencies,learning_rate,keep_prob,"
                           "batch_size,max_training_iters,"
                           "lstm_name,lstm_units,lstm_num_layers,ensemble_lstm,"
                           "wavelet_function,use_amplitude,two_dense_layers,dense_units,"
                           "decay_dense_net,vgg_num_layers,vgg_kernel_size,vgg_initial_filters,"
                           "add_l2,weight_decay,standardize_method,"
                           "val_loss,val_acc,test_loss,test_acc,"
                           "running_time,notes,date,timestamp")
                file.write("\n")

        # save results
        with open(result_path, 'a') as file:
            file.write(",".join([str(i) for i in [
                args.ahead_step, "model_" + str(args.model_structure), args.time_window,
                args.num_frequencies, args.learning_rate, args.keep_prob,
                args.batch_size, args.max_training_iters,
                args.lstm_name, args.lstm_units, args.lstm_num_layers, args.ensemble_lstm,
                args.wavelet_function, args.use_amplitude, "true" if args.two_dense_layers else "false",
                args.dense_units, "true" if args.decay_dense_net else "false",
                args.vgg_num_layers, args.vgg_kernel_size, args.vgg_initial_filters,
                "true" if args.add_l2 else "false", args.weight_decay, args.standardize_method,
                best_val_loss, best_val_acc, test_loss, test_acc,
                time.time() - time_start, notes, today, model_timestamp
            ]]))
            file.write("\n")

        logger.info("It takes {:.2f} seconds to train, validate and test.".format(time.time() - time_start))

    logger.info("done!")


