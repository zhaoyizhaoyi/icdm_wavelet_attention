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
from sklearn.metrics import mean_squared_error, mean_absolute_error
sys.path.append("../")
from core import WANN

# configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class PowerData(object):
    """
    Read stock price data from data directory and reshape them to what we need.
    """
    def __init__(self, ahead_step, train_percentage, val_percentage, time_window,
                 num_classes, num_frequencies, batch_size, model_structure=1, num_slave=20, wavelet_function="cmor",
                 training_slice=0, num_wavelet_channels=2, use_amplitude=0, standardize_method=""):
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

        self._split_data()

        # check to see whether wavelet transform has been computed already
        self.wavelet_x = None
        if self.model_structure not in [1, 5]:
            self.save_load_wavelet()

    @staticmethod
    def _load_data():
        data = pd.read_csv("../data/power/household_power_consumption.txt", sep=";")
        data = data[["Date", "Time", "Voltage"]]
        data = data[(data["Voltage"] != "?")]
        # data.reset_index(drop=True, inplace=True)
        data["Voltage"] = data["Voltage"].astype("float")
        # data["voltage_shifted"] = data["Voltage"].shift(1)
        # data = data[~(data["voltage_shifted"].isnull())]
        # data["voltage_change"] = data["voltage_shifted"] / data["Voltage"] - 1.0
        # data = data.loc[:100000]  # sample first 100,000 records for testing
        data = data[(data["Date"].str.contains("2010"))]
        data.reset_index(drop=True, inplace=True)
        return np.array(data["Voltage"])

    def _generate_tuple_list(self, index):
        j_item_tuple_list = []
        for j in range(len(self.data)):
            for item in index:
                j_item_tuple_list.append((j, item))
        return j_item_tuple_list

    def _split_data(self):
        # maybe we should write it like this
        train_pos = int(self.train_percentage * self.data.shape[0])
        val_pos = int(self.val_percentage * self.data.shape[0]) + train_pos

        self.train_index = list(range(0, train_pos - self.time_window - self.ahead_step + 1))
        self.val_index = list(range(train_pos - self.time_window - self.ahead_step + 1, val_pos - self.time_window - self.ahead_step + 1))
        self.test_index = list(range(val_pos - self.time_window - self.ahead_step + 1, self.data.shape[0] - self.time_window - self.ahead_step))

        # data normalization
        self.max_data = np.max(self.data)
        self.min_data = np.min(self.data)
        self.normalized_data = (self.data - self.min_data) / (self.max_data - self.min_data)

        self.train_size = len(self.train_index)
        self.val_size = len(self.val_index)
        self.test_size = len(self.test_index)

        logger.info(
            "\nSplit data finished!\n"
            "train: {}, val: {}, test: {}".format(
                self.train_size, self.val_size, self.test_size
            )
        )

    def _generate_time_frequency(self, time_series):
        wavelet_transformed, freqs = pywt.cwt(time_series, range(1, self.num_frequencies + 1), self.wavelet_function)
        return wavelet_transformed

    def _generate_batch_single(self, index):
        index_size = len(index)

        # the shape of batch_x, label
        batch_x = np.zeros([index_size, self.time_window, self.num_frequencies, self.num_wavelet_channels])
        label = np.zeros([index_size, self.num_classes])

        temp = 0

        for item in index:
            wavelet_transformed = self._generate_time_frequency(self.normalized_data[item:item + self.time_window])
            if self.num_wavelet_channels > 1:
                if self.use_amplitude == 1:
                    batch_x[temp, :, :] = np.stack((np.absolute(wavelet_transformed), np.arctan(wavelet_transformed.imag / wavelet_transformed.real)), axis=-1)
                else:
                    batch_x[temp, :, :] = np.stack((wavelet_transformed.real, wavelet_transformed.imag), axis=-1)
            else:
                batch_x[temp, :, :] = wavelet_transformed

            label[temp, 0] = self.normalized_data[item + self.time_window + self.ahead_step - 1]
            temp += 1
        result_tuple = [index, batch_x]
        return result_tuple

    @staticmethod
    def _split_list(parent_list, num):
        avg = len(parent_list) / float(num)
        child_list_list = []
        last = 0.0
        while last < len(parent_list):
            child_list_list.append(parent_list[int(last):int(last + avg)])
            last += avg
        return child_list_list

    def _pre_compute_save(self):
        time_pre_compute = time.time()
        index = np.array(range(0, self.data.shape[0] - self.time_window - self.ahead_step))
        index_size = len(index)

        # the shape of batch_x
        batch_x = np.zeros([index_size, self.time_window, self.num_frequencies, self.num_wavelet_channels])

        index_list_list = self._split_list(index, self.num_slave)

        p = Pool(self.num_slave)
        outputs = p.map(self._generate_batch_single, index_list_list)
        p.close()
        p.join()
        for output_i in outputs:
            temp = 0
            for item in output_i[0]:
                batch_x[item] = output_i[1][temp]
                temp += 1

        logger.info("It takes {:.2f} seconds to pre compute.".format(time.time() - time_pre_compute))
        return batch_x

    def save_load_wavelet(self):
        wavelet_dir = "./../data/wavelet_pre_computed/power/"
        if not os.path.exists(wavelet_dir):
            os.makedirs(wavelet_dir)

        if self.standardize_method == "":
            wavelet_filename = "power_wavelet_" + self.wavelet_function + "_" + str(self.time_window) + "_" + str(self.num_frequencies) + "_" + str(self.use_amplitude) + ".npy"
        else:
            wavelet_filename = "power_wavelet_" + self.wavelet_function + "_" + str(self.time_window) + "_" + str(
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

        # the shape of batch_x
        batch_x = np.zeros([index_size, self.time_window, self.num_frequencies, self.num_wavelet_channels])
        batch_y = np.zeros([index_size, self.num_classes])

        index_list_list = self._split_list(index, self.num_slave)

        p = Pool(self.num_slave)
        outputs = p.map(self._generate_batch_single, index_list_list)
        p.close()
        p.join()
        for output_i in outputs:
            temp = 0
            for item in output_i[0]:
                batch_x[item] = output_i[1][temp]
                batch_y[item] = self.normalized_data[item + self.time_window + self.ahead_step - 1]
                temp += 1
        return batch_x, batch_y

    def _generate_batch_from_pre_compute(self, index):
        index_size = len(index)

        # the shape of batch_x, label
        batch_x = np.zeros([index_size, self.time_window, self.num_frequencies, self.num_wavelet_channels])
        label = np.zeros([index_size, self.num_classes])

        temp = 0
        for item in index:
            batch_x[temp, :, :] = self.wavelet_x[item]
            label[temp, 0] = self.normalized_data[item + self.time_window + self.ahead_step - 1]
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
            batch_x, label = self._generate_batch_from_pre_compute(index)
        return batch_x, label

    def validation(self):
        index = np.array(self.val_index)
        # logger.info("index max: {}, min: {}".format(np.max(index), np.min(index)))
        # logger.info("max y pos: {}".format(np.max(index) + self.time_window + self.ahead_step - 1))
        if self.wavelet_x is None:
            batch_x, label = self._generate_batch(index)
        else:
            batch_x, label = self._generate_batch_from_pre_compute(index)
        return batch_x, label

    def testing(self):
        index = np.array(self.test_index)
        # logger.info("index max: {}, min: {}".format(np.max(index), np.min(index)))
        # logger.info("max y pos: {}".format(np.max(index) + self.time_window + self.ahead_step - 1))
        if self.wavelet_x is None:
            batch_x, label = self._generate_batch(index)
        else:
            batch_x, label = self._generate_batch_from_pre_compute(index)
        return batch_x, label

    def _generate_batch_lstm(self, index):
        index_size = len(index)

        # the shape of batch_x, label
        batch_x = np.zeros([index_size, self.time_window, 1])
        label = np.zeros([index_size, self.num_classes])

        temp = 0
        for item in index:
            batch_x[temp, :, :] = np.reshape(self.normalized_data[item:item + self.time_window], (-1, 1))
            label[temp, 0] = self.normalized_data[item + self.time_window + self.ahead_step - 1]
            temp += 1
        return batch_x, label

    def next_batch_lstm(self):
        index = random.sample(self.train_index, self.batch_size)
        index = np.array(index)
        batch_x, label = self._generate_batch_lstm(index)
        return batch_x, label

    def validation_lstm(self):
        index = np.array(self.val_index)
        batch_x, label = self._generate_batch_lstm(index)
        return batch_x, label

    def testing_lstm(self):
        index = np.array(self.test_index)
        batch_x, label = self._generate_batch_lstm(index)
        return batch_x, label

    def next_batch_combined(self):
        """We make this next_batch_combined single, because it involves a random process.
        Therefore, we cannot make the batch_x and label the same unless we provide the same index
        to generate_batch method.
        """
        index = random.sample(self.train_index, self.batch_size)
        index = np.array(index)

        # for lstm
        batch_x_lstm, label_lstm = self._generate_batch_lstm(index)

        # for wavelet
        if self.wavelet_x is None:
            batch_x_wavelet, label_wavelet = self._generate_batch(index)
        else:
            batch_x_wavelet, label_wavelet = self._generate_batch_from_pre_compute(index)

        return {
            "batch_x_lstm": batch_x_lstm,
            "batch_x_wavelet": batch_x_wavelet,
            "label": label_lstm
        }


if __name__ == "__main__":
    today = datetime.date.today().strftime("%Y%m%d")
    logger.info("Running model on {}.".format(today))
    # time.sleep(random.uniform(1, 2))
    # for looking checkpoints and log
    model_timestamp = int(time.time() * 1000 + int(os.getpid()))
    logger.info("Model timestamp: {}".format(str(model_timestamp)))

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
    parser.add_argument("--export_prediction_results",
                        help="export the prediction results or not",
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
        power_data = PowerData(
            ahead_step=args.ahead_step,
            train_percentage=0.8,
            val_percentage=0.1,
            time_window=args.time_window,
            num_classes=1,
            num_frequencies=args.num_frequencies,
            batch_size=args.batch_size,
            model_structure=args.model_structure,
            wavelet_function=args.wavelet_function,
            training_slice=args.training_slice,
            num_wavelet_channels=args.num_wavelet_channels,
            use_amplitude=args.use_amplitude,
            standardize_method=args.standardize_method
        )

        # # for looking checkpoints and log
        # model_timestamp = int(time.time() * 1000)
        # logger.info("Model timestamp: {}".format(str(model_timestamp)))

        cnn_model = WANN(
            sess=sess,
            ahead_step=args.ahead_step,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            keep_prob=args.keep_prob,
            time_window=args.time_window,
            num_channels=args.num_frequencies,
            num_classes=1,
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

        test_pred, test_y, best_val_loss, test_loss = cnn_model.train(power_data)

        # calculate mse, rmse, mae and mape
        y_test = test_y * (power_data.max_data - power_data.min_data) + power_data.min_data
        y_pred = test_pred * (power_data.max_data - power_data.min_data) + power_data.min_data

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        logger.info("MSE: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}".format(
            mse, rmse, mae, mape
        ))

        # calculate mse for naive prediction
        naive_test = y_test[args.ahead_step:]
        naive_pred = y_test[:-1 * args.ahead_step]
        mse_naive = mean_squared_error(naive_test, naive_pred)
        rmse_naive = np.sqrt(mse_naive)
        mae_naive = mean_absolute_error(naive_test, naive_pred)
        mape_naive = mean_absolute_percentage_error(naive_test, naive_pred)
        logger.info("Naive MSE: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}".format(
            mse_naive, rmse_naive, mae_naive, mape_naive
        ))

        # wavelet function
        notes = args.notes + "_wavelet_" + args.wavelet_function

        # specify results path
        result_dir = "../results/power/"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_path = os.path.join(result_dir, 'power_results_step_' + str(args.ahead_step) + ".csv")

        if not os.path.exists(result_path):
            with open(result_path, 'a') as file:
                file.write("ahead_step,model_structure,time_window,num_frequencies,learning_rate,keep_prob,"
                           "batch_size,max_training_iters,"
                           "lstm_name,lstm_units,lstm_num_layers,ensemble_lstm,"
                           "wavelet_function,use_amplitude,two_dense_layers,dense_units,"
                           "decay_dense_net,vgg_num_layers,vgg_kernel_size,vgg_initial_filters,"
                           "add_l2,weight_decay,standardize_method,"
                           "mse_naive,rmse_naive,mae_naive,mape_naive,"
                           "val_loss,test_loss,mse,rmse,mae,mape,"
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
                mse_naive, rmse_naive, mae_naive, mape_naive,
                best_val_loss, test_loss, mse, rmse, mae, mape,
                time.time() - time_start, notes, today, model_timestamp
            ]]))
            file.write("\n")

        if args.export_prediction_results:
            prediction_results = pd.DataFrame(columns=["y_test", "y_pred", "error_" + str(args.model_structure) + "_step"])
            prediction_results["y_test"] = np.ravel(y_test)
            prediction_results["y_pred"] = np.ravel(y_pred)
            prediction_results["error_" + str(args.model_structure) + "_step"] = np.abs(prediction_results["y_pred"] - prediction_results["y_test"])
            prediction_dir = "../data/prediction/power/"
            if not os.path.exists(prediction_dir):
                os.makedirs(prediction_dir)
            prediction_file = os.path.join(prediction_dir, "power_prediction_" + str(args.ahead_step) + "_" + str(args.model_structure) + ".csv")
            prediction_results.to_csv(prediction_file, index=False, header=True, encoding="utf-8")

        logger.info("It takes {:.2f} seconds to train, validate and test.".format(time.time() - time_start))
    logger.info("done!")


