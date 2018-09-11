# icdm_wavelet_attention
*combine wavelet transform and attention mechanism 
for time series forecasting or classification*

This code is implementation of "Forecasting Wavelet Transformed Time Series with Attentive Neural Networks" (ICDM 2018).
The models are defined in the `core` directory. 
The experiments on two datasets are defined in `power` and `stock` directories, respectively.

The hyper-parameters could be set as arguments to the scripts like this: <br />
```python stock_data.py --ahead_step=1 --time_window=5 --num_frequencies=5 --lstm_units=8 --max_training_iters=50 --keep_prob=1.0 --model_structure=1 --notes=pure_lstm --learning_rate=0.01```
where `model_structure` determines which model you choose:
+ model_structure = 1: LSTM;
+ model_structure = 2: CNN;
+ model_structure = 3: Our attentive neural network;
+ model_structure = 4: ensemble of LSTM and CNN.

The data could be downloaded according to the website provided in the paper. 


