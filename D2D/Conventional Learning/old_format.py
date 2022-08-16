import os
import datetime
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import pickle
from tensorflow import keras
from keras.callbacks import History
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_absolute_error
from WindowGenerator2 import WindowGenerator
import random as python_random
import argparse


np.random.seed(123)
python_random.seed(123)
tf.random.set_seed(123)

# =====================================================================
#          PARSE parameters from shell file or terminal script
# =====================================================================

parser = argparse.ArgumentParser() #create parser object
#define each argument / input

parser.add_argument("--input_width", help="specify input data width", type=int) 
parser.add_argument("--out_steps", help="specify number of steps in the future to predict", type=int) 
parser.add_argument("--shift", help="specify offset", type=int) 
args = parser.parse_args()
# args, unknown = parser.parse_known_args() 

# # ==================================================
# #              Define program variables
# # ==================================================

# # when specifying zero vaules, the default will be set 
# # A 'hard coded' varible will need set to set zero values

# LSTM_SIZE_1
# if args.lstm_size_1:
#     print('Specified LSTM_SIZE_1:', args.lstm_size_1)
#     LSTM_SIZE_1 = args.lstm_size_1
# else:
#     print('No specified LSTM_SIZE_1 - default: 50')
#     LSTM_SIZE_1 = 50 #50,100,200

# # LSTM_SIZE_2
# if args.lstm_size_2:
#     print('Specified LSTM_SIZE_2:', args.lstm_size_2)
#     LSTM_SIZE_2 = args.lstm_size_2
# else:
#     print('No specified LSTM_SIZE_2 - default: 0')
#     LSTM_SIZE_2 = 50 #50,100,200

# INPUT_WIDTH
if args.input_width:
    print('Specified INPUT_WIDTH:', args.input_width)
    INPUT_WIDTH = args.input_width
else:
    print('No specified INPUT_WIDTH - default: 30')
    INPUT_WIDTH = 25

# OUT_STEPS
if args.out_steps:
    print('Specified OUT_STEPS:', args.out_steps)
    OUT_STEPS = args.out_steps
else:
    print('No specified OUT_STEPS - default: 15')
    OUT_STEPS = 4



close_plots = False
show_plots = True
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)

## HYPERPARAMETERS
# data_path = "data/fast_fading_dB_LOS_Head_Indoor_downsampled100hz_n50.txt"
data_path = "pathAB_SSF_dB_AP1_downsampled2Khz_win100.txt"
# TRAIN_STARTINDEX = 0
# TEST_ENDINDEX = 18113
MODEL = 'LSTM'
config_layer_1 = [50, 100, 200] # hidden units layer 1
# config_layer_2 = [1, 5, 10, 25, 50, 100, 200, 500] # hidden units layer 2
config_layer_2 = [] # declare this variable as an empty list for one-layer model
# INPUT_WIDTH = 17
# OUT_STEPS = 33
SHIFT = OUT_STEPS
# SHIFT = 10
MAX_EPOCHS = 50


## DON'T CHANGE
BATCHSIZE = 32
DROPOUT = 0.3

# Types of normalization/scaling implemented here
# NORM = 0 #standardization
# NORM = 1 #centred mean and min = -1
# NORM = 2 #minmax

# Chosen min-max normalisation for the paper
NORM = 2

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# DataFrame data is dealt as a column, so we must transpose the returned row data via ".T"
# df = pd.read_csv(data_path, header=None, delimiter=r"\s+").T
df = pd.read_csv(data_path, header=None, delimiter=r"\s+")
df = df.rename(columns={0: "fast-fading (dB)"})
df.head()
print(df)
# %%
plot_cols = 'fast-fading (dB)'
plot_features = df[plot_cols]
_ = plot_features.plot()
plt.ylabel('fast-fading (dB)')
plt.xlabel('Sample number')

if show_plots:
  plt.show()
if close_plots:
  plt.close()

df.describe().transpose()

column_indices = {name: i for i, name in enumerate(df.columns)}

# n = len(df[TRAIN_STARTINDEX:TEST_ENDINDEX])
# train_df = df[TRAIN_STARTINDEX:int(n*0.7)]
# val_df = df[int(n*0.7):int(n*0.9)]
# test_df = df[int(n*0.9):TEST_ENDINDEX]

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]

train_mean = train_df.mean()
train_std = train_df.std()
train_min = train_df.min()
train_max = train_df.max()

print('train_min', train_min)
print('train_max', train_max)

#standardization
if NORM == 0: 
  
  train_df = (train_df - train_mean) / train_std
  val_df = (val_df - train_mean) / train_std
  test_df = (test_df - train_mean) / train_std
  
  print(train_df.mean())
  print(train_df.max())
  print(train_df.min())

  print(val_df.mean())
  print(test_df.mean())

  # Now peek at the distribution of the features. 
  # Some features do have long tails, but there are no obvious errors.

  df_std = (df - train_mean) / train_std
  df_std = df_std.melt(var_name='Column', value_name='Standardization')
  plt.figure(figsize=(12, 6))
  ax = sns.violinplot(x='Column', y='Standardization', data=df_std)
  _ = ax.set_xticklabels(df.keys(), rotation=90)

  if show_plots:
    plt.show()
  if close_plots:
    plt.close()

  plot_cols = 'Standardization'
  plot_features = df_std[plot_cols]
  _ = plot_features.plot()
  plt.ylabel('fast-fading (dB) [normed]')
  plt.xlabel('Sample number')

  if show_plots:
    plt.show()
  if close_plots:
    plt.close()

#centred mean and min value = -1 and max value < 1
if NORM == 1: 
  
  maxmean = abs(train_max - train_mean)
  minmean = abs(train_min - train_mean)

  max1 = abs(train_max)
  min1 = abs(train_min)
  
  train_df = (train_df - train_mean) / (max(maxmean[(0)], minmean[(0)]))
  val_df = (val_df - train_mean) / (max(maxmean[(0)], minmean[(0)]))
  test_df = (test_df - train_mean) / (max(maxmean[(0)], minmean[(0)]))

  # Now peek at the distribution of the features. 
  # Some features do have long tails, but there are no obvious errors.
  df_norm = (df - train_mean) / (max(maxmean[(0)], minmean[(0)]))
  df_norm = df_norm.melt(var_name='Column', value_name='Normalized')
  plt.figure(figsize=(12, 6))
  ax = sns.violinplot(x='Column', y='Normalized', data=df_norm)
  _ = ax.set_xticklabels(df.keys(), rotation=90)
  
  if show_plots:
    plt.show()
  if close_plots:
    plt.close()

  plot_cols = 'Normalized'
  plot_features = df_norm[plot_cols]
  _ = plot_features.plot()
  plt.ylabel('fast-fading (dB) [normed]')
  plt.xlabel('Sample number')
  
  if show_plots:
    plt.show()
  if close_plots:
    plt.close()


#minmax [-1,1] normaliztion
if NORM == 2: 
  # scaler = MinMaxScaler(feature_range=(-1,1))
  # scaled = scaler.fit_transform(train_df)
  print(train_df)

  train_df = (2*(train_df - train_min) / (train_max - train_min)) -1
  val_df = (2*(val_df - train_min) / (train_max - train_min))-1
  #test_df = train_min[0] + (train_max[0] - train_min[0]) * (train_df + 1)/2
  test_df = (2*(test_df - train_min) / (train_max - train_min))-1

  # Now peek at the distribution of the features. Some features do have long tails, but there are no obvious errors.

  df_minmax = (2 * (df - train_min)/(train_max - train_min)) - 1 
  df_minmax = df_minmax.melt(var_name='Column', value_name='MAX-MIN Normalized')
  plt.figure(figsize=(12, 6))
  ax = sns.violinplot(x='Column', y='MAX-MIN Normalized', data=df_minmax)
  _ = ax.set_xticklabels(df.keys(), rotation=90)

  if show_plots:
    plt.show()
  if close_plots:
    plt.close()

  plot_cols = 'MAX-MIN Normalized'
  plot_features = df_minmax[plot_cols]
  _ = plot_features.plot()
  plt.ylabel('fast-fading (dB) [normed]')
  plt.xlabel('Sample number')

  if show_plots:
    plt.show()
  if close_plots:
    plt.close()

def compile_and_fit(model, window, patience=10, file_name="models", save_weights_only=False):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min',
                                                    restore_best_weights=True)
  
  reduceLrOnPlateau = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                           patience=5,
                                                           verbose=1)

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError(),
                         tf.metrics.RootMeanSquaredError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      verbose = 0,
                      callbacks=[early_stopping, reduceLrOnPlateau])

  print(history.history)
  
  # if save_weights_only:
  #   model.save_weights(model_path)
  # else:
  #   model.save(file_name)
   
  # f = open(os.path.join(file_name,'losses.pckl'),'wb')
  # pickle.dump(history.history, f)
  # f.close()

  return history

# %%
# invert min_max normalization
def min_max_inverse(data):
  # print(train_min[0])
  # print(train_max[0])
  # print(data)
  # print(train_min[0] + (train_max[0] - train_min[0]) * (data + 1)/2)
  return train_min[0] + (train_max[0] - train_min[0]) * (data + 1)/2


multi_window = WindowGenerator(input_width=INPUT_WIDTH,
                               label_width=OUT_STEPS,
                               shift=SHIFT,
                               train_df=train_df,
                               val_df=val_df,
                               test_df=test_df,
                               inverse_function=min_max_inverse)

multi_window.plot()
plt.suptitle('Multi window plot with inputs and labels')
multi_window

# %%time

# figures_dir = '/lstm_input_'+str(INPUT_WIDTH)+'_output_'+str(OUT_STEPS)
figures_dir = '/home/nidhisimmons/git/D2D_channel_prediction_Sept2021/mmwave_lstm_input_'+str(INPUT_WIDTH)+'_output_'+str(OUT_STEPS)

if not os.path.isdir(figures_dir):
       os.makedirs(figures_dir)

val_mse_errors = []
val_rmse_errors = [] 
val_mae_errors = []
test_mse_errors = []
test_rmse_errors = [] 
test_mae_errors = []

val_mse_errors_reversescaled = []
val_rmse_errors_reversescaled = [] 
val_mae_errors_reversescaled = []
test_mse_errors_reversescaled = []
test_rmse_errors_reversescaled = [] 
test_mae_errors_reversescaled = []

for i in range(len(config_layer_1)):
  if len(config_layer_2) > 0:
    print('##### {} MODEL WITH {} - {} UNITS'.format(MODEL, config_layer_1[i], config_layer_2[i]))
    lstm_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, lstm_units]
        # Adding more `lstm_units` just overfits more quickly.
        tf.keras.layers.LSTM(config_layer_1[i], return_sequences=True),
        # Adding dropout to prent overfitting
        tf.keras.layers.Dropout(DROPOUT),
        # second lstm layer
        tf.keras.layers.LSTM(config_layer_2[i], return_sequences=False),
        # # # Adding dropout to prent overfitting
        tf.keras.layers.Dropout(DROPOUT),
        # Shape => [batch, out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS*multi_window.get_num_features()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, multi_window.get_num_features()])
    ])
  else:
    print('##### {} MODEL WITH {} UNITS'.format(MODEL, config_layer_1[i]))
    lstm_model = tf.keras.Sequential([                         
        # Shape [batch, time, features] => [batch, lstm_units]
        # Adding more `lstm_units` just overfits more quickly.
        tf.keras.layers.LSTM(config_layer_1[i], return_sequences=False),
        # Adding dropout to prent overfitting
        tf.keras.layers.Dropout(DROPOUT),
        # Shape => [batch, out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS*multi_window.get_num_features()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, multi_window.get_num_features()])
      ])
  history_lstm = compile_and_fit(lstm_model, multi_window)

  multi_val_performance = {}
  multi_val_performance_verification = {}
  multi_test_performance = {}
  multi_test_performance_verification = {}

  # IPython.display.clear_output()

  lstm_loss_train = history_lstm.history['loss']
  lstm_loss_val = history_lstm.history['val_loss']
  lstm_epochs =  range(len(lstm_loss_train))

  multi_val_performance_rt_lstm = {}
  multi_test_performance_rt_lstm = {}

  #validation dataset
  mae_val = 0
  mse_val = 0
  rmse_val = 0

  mae_val_reversescaled = 0
  mse_val_reversescaled = 0
  rmse_val_reversescaled = 0

  b = 0
  val_true = 0
  val_pred = 0
  diff_val = 0
  diff_val_reversescaled = 0

  #Using model.predict() to obtain the same result as model.evalute()
  for b in iter(multi_window.val):

    #reverse transform the val_true and val_pred and then compute the metrics
    val_true = min_max_inverse(b[1]) #val_true or val_labels
    val_pred = min_max_inverse(lstm_model.predict(b[0]))
  
    diff_val = b[1] - lstm_model.predict(b[0]) #val_labels (or val_true) - val_pred; scaled version
    diff_val_reversescaled = val_true - val_pred #val_labels (or val_true) - val_pred: reverse-scaled

    mae_val += np.mean(abs(diff_val))
    mse_val += np.mean(diff_val**2)

    mae_val_reversescaled += np.mean(abs(diff_val_reversescaled))
    mse_val_reversescaled += np.mean(diff_val_reversescaled**2)
    
  mae_val/=(len(multi_window.val)-1)+(len(diff_val))/BATCHSIZE #(106+15/32)
  mse_val/=(len(multi_window.val)-1)+(len(diff_val))/BATCHSIZE #(106+15/32)
  rmse_val = np.sqrt(mse_val)

  mae_val_reversescaled/=(len(multi_window.val)-1)+(len(diff_val_reversescaled))/BATCHSIZE #(106+15/32)
  mse_val_reversescaled/=(len(multi_window.val)-1)+(len(diff_val_reversescaled))/BATCHSIZE #(106+15/32)
  rmse_val_reversescaled = np.sqrt(mse_val_reversescaled)

  multi_val_performance_rt_lstm[1] = mae_val_reversescaled
  multi_val_performance_rt_lstm[2] = rmse_val_reversescaled

  multi_val_performance['LSTM'] = lstm_model.evaluate(multi_window.val)


  #test dataset
  mae_test = 0
  mse_test = 0
  rmse_test = 0

  mae_test_reversescaled = 0
  mse_test_reversescaled = 0
  rmse_test_reversescaled = 0

  c = 0
  test_true = 0
  test_pred = 0
  diff_test = 0
  diff_test_reversescaled = 0

  #Using model.predict() to obtain the same result as model.evalute()
  for c in iter(multi_window.test):

    #reverse transform test_true and test_pred and then compute the metrics
    test_true = min_max_inverse(c[1]) #test_labels
    test_pred = min_max_inverse(lstm_model.predict(c[0]))
  
    diff_test = c[1] - lstm_model.predict(c[0]) # test_labels (or test_true) - test_pred
    diff_test_reversescaled = test_true - test_pred #test_labels (or test_true) - test_pred: reverse-scaled

    mae_test += np.mean(abs(diff_test))
    mse_test += np.mean(diff_test**2)

    mae_test_reversescaled += np.mean(abs(diff_test_reversescaled))
    mse_test_reversescaled += np.mean(diff_test_reversescaled**2)

  mae_test /= (len(multi_window.test)-1)+(len(diff_test))/BATCHSIZE #(49+28/32)
  mse_test /= (len(multi_window.test)-1)+(len(diff_test))/BATCHSIZE #(49+28/32)
  rmse_test = np.sqrt(mse_test)

  mae_test_reversescaled/=(len(multi_window.test)-1)+(len(diff_test_reversescaled))/BATCHSIZE #(106+15/32)
  mse_test_reversescaled/=(len(multi_window.test)-1)+(len(diff_test_reversescaled))/BATCHSIZE #(106+15/32)
  rmse_test_reversescaled = np.sqrt(mse_test_reversescaled)

  multi_test_performance_rt_lstm[1] = mae_test_reversescaled
  multi_test_performance_rt_lstm[2] = rmse_test_reversescaled

  multi_test_performance['LSTM'] = lstm_model.evaluate(multi_window.test, verbose=0)

  multi_window.plot(lstm_model)
  if len(config_layer_2) > 0:
    plt.suptitle('Model: {} | Hidden Units: {} - {} | Input: {} | Output: {}'.format(MODEL, config_layer_1[i], config_layer_2[i], INPUT_WIDTH, OUT_STEPS))
  else:
    plt.suptitle('Model: {} | Hidden Units: {} | Input: {} | Output: {}'.format(MODEL, config_layer_1[i], INPUT_WIDTH, OUT_STEPS))
  # plt.savefig(model_directory+"/multi_dense")

  val_mse_errors.append(multi_val_performance['LSTM'][0])
  val_mae_errors.append(multi_val_performance['LSTM'][1]) 
  val_rmse_errors.append(multi_val_performance['LSTM'][2])
  test_mse_errors.append(multi_test_performance['LSTM'][0])
  test_mae_errors.append(multi_test_performance['LSTM'][1]) 
  test_rmse_errors.append(multi_test_performance['LSTM'][2])


  val_rmse_errors_reversescaled.append(rmse_val_reversescaled)
  val_mae_errors_reversescaled.append(mae_val_reversescaled)

  test_rmse_errors_reversescaled.append(rmse_test_reversescaled)
  test_mae_errors_reversescaled.append(mae_test_reversescaled)


  if show_plots:
    plt.show()
  if close_plots:
    plt.close()

  plt.plot(lstm_epochs, lstm_loss_train, 'g', label='Training loss')
  plt.plot(lstm_epochs, lstm_loss_val, 'b', label='Validation loss')
  if len(config_layer_2) > 0:
    plt.title('{} Model Loss | Hidden Units: {} - {} | Input: {} | Output: {}'.format(MODEL, config_layer_1[i], config_layer_2[i], INPUT_WIDTH, OUT_STEPS))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(figures_dir+"/losses_lstm_"+str(config_layer_1[i])+'_'+str(config_layer_2[i]))
  else:
    plt.title('{} Model Loss | Hidden Units: {} | Input: {} | Output: {}'.format(MODEL, config_layer_1[i], INPUT_WIDTH, OUT_STEPS))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(figures_dir+"/losses_lstm_"+str(config_layer_1[i]))


  if show_plots:
    plt.show()
  if close_plots:
    plt.close()

x = np.arange(len(test_rmse_errors))
width = 0.2

plt.bar(x - 0.17, val_mae_errors, width, label='Validation')
plt.bar(x + 0.17, test_mae_errors, width, label='Test')
# plt.xticks(ticks=x, labels = [str(x) + ' units' for x in config],
#            rotation=45)
if len(config_layer_2) > 0:
  plt.xticks(ticks=x, labels = ['{}-{} units'.format(x,y) for x,y in zip(config_layer_1, config_layer_2)],
           rotation=40)
else:
  plt.xticks(ticks=x, labels = [str(x) + ' units' for x in config_layer_1],
           rotation=45)
plt.ylabel(f'Mean Absolute Error (MAE)')
_ = plt.legend()
plt.title('{} - Hyperparameters Search - Input: {} | Outuput: {}'.format(MODEL, INPUT_WIDTH, OUT_STEPS))
plt.savefig(figures_dir+"/maes_lstm")
if show_plots:
    plt.show()
if close_plots:
    plt.close()

plt.bar(x - 0.17, val_rmse_errors, width, label='Validation')
plt.bar(x + 0.17, test_rmse_errors, width, label='Test')
# plt.xticks(ticks=x, labels = [str(x) + ' units' for x in config],
#            rotation=45)
if len(config_layer_2) > 0:
  plt.xticks(ticks=x, labels = ['{}-{} units'.format(x,y) for x,y in zip(config_layer_1, config_layer_2)],
           rotation=40)
else:
  plt.xticks(ticks=x, labels = [str(x) + ' units' for x in config_layer_1],
           rotation=45)
plt.ylabel(f'Root Mean Square Error (RMSE)')
_ = plt.legend()
plt.title('{} - Hyperparameters Search - Input: {} | Outuput: {}'.format(MODEL, INPUT_WIDTH, OUT_STEPS))
plt.savefig(figures_dir+"/rmses_lstm")
if show_plots:
    plt.show()
if close_plots:
    plt.close()

plt.bar(x - 0.17, val_mae_errors_reversescaled, width, label='Validation')
plt.bar(x + 0.17, test_mae_errors_reversescaled, width, label='Test')
# plt.xticks(ticks=x, labels = [str(x) + ' units' for x in config],
#            rotation=45)
if len(config_layer_2) > 0:
  plt.xticks(ticks=x, labels = ['{}-{} units'.format(x,y) for x,y in zip(config_layer_1, config_layer_2)],
           rotation=40)
else:
  plt.xticks(ticks=x, labels = [str(x) + ' units' for x in config_layer_1],
           rotation=45)
plt.ylabel(f'Mean Absolute Error (MAE) - REVERSESCALED')
_ = plt.legend()
plt.title('{} - Hyperparameters Search - Input: {} | Outuput: {}'.format(MODEL, INPUT_WIDTH, OUT_STEPS))
plt.savefig(figures_dir+"/maes_lstm_RT")
if show_plots:
    plt.show()
if close_plots:
    plt.close()

plt.bar(x - 0.17, val_rmse_errors_reversescaled, width, label='Validation')
plt.bar(x + 0.17, test_rmse_errors_reversescaled, width, label='Test')
# plt.xticks(ticks=x, labels = [str(x) + ' units' for x in config],
#            rotation=45)
if len(config_layer_2) > 0:
  plt.xticks(ticks=x, labels = ['{}-{} units'.format(x,y) for x,y in zip(config_layer_1, config_layer_2)],
           rotation=40)
else:
  plt.xticks(ticks=x, labels = [str(x) + ' units' for x in config_layer_1],
           rotation=45)
plt.ylabel(f'Root Mean Squared Error (RMSE) - REVERSESCALED')
_ = plt.legend()
plt.title('{} - Hyperparameters Search - Input: {} | Outuput: {}'.format(MODEL, INPUT_WIDTH, OUT_STEPS))
plt.savefig(figures_dir+"/rmses_lstm_RT")
if show_plots:
    plt.show()
if close_plots:
    plt.close()

dt = {'MAE': [x for x in test_mae_errors],
      'RMSE': [x for x in test_rmse_errors]}

# df = pd.DataFrame(dt, index=[str(x) + ' units' for x in config])
if len(config_layer_2) > 0:
  df = pd.DataFrame(dt, index=['{} - {} units'.format(x,y) for x,y in zip(config_layer_1, config_layer_2)])
else:
  df = pd.DataFrame(dt, index=[str(x) + ' units' for x in config_layer_1])

print('{} - Errors Comparison'.format(MODEL))
print(df)

with open(figures_dir+'/errors.txt', 'a') as f:
    f.write('{} - Errors Comparison\n'.format(MODEL))
    dfAsString = df.to_string(header=True, index=True)
    f.write(dfAsString)

dt = {'MAE': [x for x in test_mae_errors_reversescaled],
      'RMSE': [x for x in test_rmse_errors_reversescaled]}

# df = pd.DataFrame(dt, index=[str(x) + ' units' for x in config])
if len(config_layer_2) > 0:
  df = pd.DataFrame(dt, index=['{} - {} units'.format(x,y) for x,y in zip(config_layer_1, config_layer_2)])
else:
  df = pd.DataFrame(dt, index=[str(x) + ' units' for x in config_layer_1])

print('{} - Errors Comparison - REVERSESCALED'.format(MODEL))
print(df)

with open(figures_dir+'/errors_reversescaled.txt', 'a') as f:
    f.write('{} - Errors Comparison - REVERSESCALED\n'.format(MODEL))
    dfAsString = df.to_string(header=True, index=True)
    f.write(dfAsString)