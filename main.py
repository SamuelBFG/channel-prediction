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
from WindowGenerator import WindowGenerator
from AuxiliaryMethods import *
import random as python_random
import settings as s

np.random.seed(123)
python_random.seed(123)
tf.random.set_seed(123)

######################################################################################################    
########################################### HYPERPARAMETERS ##########################################
######################################################################################################

# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


######################################################################################################    
############################################## DATA LOAD #############################################
######################################################################################################

df = pd.read_csv(s.DATA_PATH, header=None, delimiter=r"\s+")
df = df.rename(columns={0: "fast-fading (dB)"})
# df.plot(figsize=(10,8));


n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

train_mean = train_df.mean()
train_std = train_df.std()
train_min = train_df.min()
train_max = train_df.max()

num_features = df.shape[1]
print('Number of features: ', num_features)


ax = train_df.plot(figsize=(10,8))
val_df.plot(ax=ax)
test_df.plot(ax=ax)
ax.legend(('Train', 'Val', 'Test'))
plt.xlabel('Sample')
plt.ylabel('Small-scale fading (dB)')

if s.SHOW_PLOTS:
  plt.show()


######################################################################################################    
######################################## DATA NORMALIZATION ##########################################
######################################################################################################

train_df, val_df, test_df = normalize_data(df, train_df, val_df, test_df)

######################################################################################################    
########################################### BUILDING WINDOW ##########################################
######################################################################################################

window = WindowGenerator(input_width=s.INPUT_WIDTH,
                               label_width=s.OUT_STEPS,
                               shift=s.OUT_STEPS,
                               train_df=train_df,
                               val_df=val_df,
                               test_df=test_df,
                               train_min=train_min,
                               train_max=train_max,
                               inverse_function=min_max_inverse)

if s.SHOW_PLOTS:
    window.plot()
    plt.suptitle('Multi window plot with inputs and labels')

######################################################################################################    
#################################### FIT LINEAR REGRESSION ###########################################
######################################################################################################

linear_models = linear_histories = [] # keras.sequential models and keras.history objects
linear_mae = linear_rmse = linear_mae_rt = linear_rmse_rt = {} # errors dicts

# fit model and and output errors
linear_models, linear_histories, \
linear_mae, linear_rmse, linear_mae_rt, linear_rmse_rt = fitLinearRegression(window, train_min, train_max)

## Plot and save (.txt) errors
plot_errors(linear_mae, linear_rmse, linear_mae_rt, linear_rmse_rt, model_name='LINEAR')

######################################################################################################    
############################################# FIT MLP ################################################
######################################################################################################
# variables
dense_models = dense_histories = [] # keras.sequential models and keras.history objects
dense_mae = dense_rmse = dense_mae_rt = dense_rmse_rt = {} # errors dicts

# fit model and and output errors
dense_models, dense_histories, \
dense_mae, dense_rmse, dense_mae_rt, dense_rmse_rt = fitMLP(window, train_min, train_max)

## Plot and save (.txt) errors
plot_errors(dense_mae, dense_rmse, dense_mae_rt, dense_rmse_rt, model_name='MLP')

######################################################################################################    
############################################# FIT LSTM ###############################################
######################################################################################################
# variables
lstm_models = lstm_histories = [] # keras.sequential models and keras.history objects
lstm_mae = lstm_rmse = lstm_mae_rt = lstm_rmse_rt = {} # errors dicts

# fit model and and output errors
lstm_models, lstm_histories, \
lstm_mae, lstm_rmse, lstm_mae_rt, lstm_rmse_rt = fitLSTM(window, train_min, train_max)

## Plot and save (.txt) errors
plot_errors(lstm_mae, lstm_rmse, lstm_mae_rt, lstm_rmse_rt, model_name='LSTM')

######################################################################################################    
############################################# FIT GRU ################################################
######################################################################################################

gru_models = gru_histories = [] # keras.sequential models and keras.history objects
gru_mae = gru_rmse = gru_mae_rt = gru_rmse_rt = {} # errors dicts

# fit model and and output errors
gru_models, gru_histories, \
gru_mae, gru_rmse, gru_mae_rt, gru_rmse_rt = fitGRU(window, train_min, train_max)

## Plot and save (.txt) errors
plot_errors(gru_mae, gru_rmse, gru_mae_rt, gru_rmse_rt, model_name='GRU')

######################################################################################################    
############################################# FIT CNN ################################################
######################################################################################################

cnn_models = cnn_histories = [] # keras.sequential models and keras.history objects
cnn_mae = cnn_rmse = cnn_mae_rt = cnn_rmse_rt = {} # errors dicts

# fit model and and output errors
cnn_models, cnn_histories, \
cnn_mae, cnn_rmse, cnn_mae_rt, cnn_rmse_rt = fitCNN(window, train_min, train_max)

## Plot and save (.txt) errors
plot_errors(cnn_mae, cnn_rmse, cnn_mae_rt, cnn_rmse_rt, model_name='1DCNN')

