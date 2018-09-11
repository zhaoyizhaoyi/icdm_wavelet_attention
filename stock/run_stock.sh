#!/bin/bash


echo "model_1: lstm"
echo "pure lstm, ahead_step: 1"
python stock_data.py --ahead_step=1 --time_window=5 --num_frequencies=5 --lstm_units=8 --max_training_iters=50 --keep_prob=1.0 --model_structure=1 --notes=pure_lstm --learning_rate=0.01


echo "done"
