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
from aux_funcs import normalize_data, plot_errors, min_max_inverse, get_report_dict
from models_fit import fitLinearRegression, fitMLP, fitLSTM, fitARLSTM, fitGRU, fitCNN, fitNBEATS
import random as python_random
import argparse
import datetime
import pdb

# np.random.seed(123)
# python_random.seed(123)
# tf.random.set_seed(123)

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
cont = 1

def parse_args():
  parser = argparse.ArgumentParser(description='wireless_channel_forecasting')

  # models I/O
  parser.add_argument("--input_width", help="specify input data width", type=int, default=25) 
  parser.add_argument("--out_steps", help="specify no. of steps in the future to predict", type=int, default=50) 
  parser.add_argument("--shift", help="specify offset", type=int, default=None)

  # neural networks architecture hyperparameters
  parser.add_argument('--num_units_l1', type=list, default=[32, 64, 128, 256, 512, 1024], help='no. of units in hidden layer 1')
  parser.add_argument('--num_units_l2', type=list, default=[], help='no. of units in hidden layer 2')
  parser.add_argument('--conv_width', type=int, default=5, help='kernel size for convolutional layers')
  parser.add_argument('--max_epochs', type=int, default=100, help='no. of training epochs')
  parser.add_argument('--dropout_rate', type=float, default=0, help='dropout rate (default 30%)')

  # n-beats architecture
  parser.add_argument('--n_layers', type=int, default=3, help='no. of layers for each fully-connected network (N-BEATS)')
  parser.add_argument('--n_blocks', type=int, default=3, help='no. of blocks for the generic architecture (N-BEATS)')
  parser.add_argument('--b_sharing', dest='b_sharing', action='store_false', default=True,
                       help='True to share weights between the blocks (N-BEATS)')

  # config
  parser.add_argument('--mb_size', type=int, default=32, help='minibatch size')
  parser.add_argument('--norm_type', type=int, default=0, help='data normalization type (0 for standardization, \
                                                                1 for centred mean and min = -1, 2 for minmax)')
  parser.add_argument('--n_trials', type=int, default=10, help='no. of trials')
  parser.add_argument('--if_show_plots', dest='if_show_plots', action='store_true', default=False, help='True to show plots')

  # paths
  parser.add_argument('--data_path', dest='data_path', type=str, default='fast_fading_dB_1kHz_indoor_nlos.txt',
                       help='raw dataset path')
  parser.add_argument('--path_for_common_dir', type=str, dest='path_for_common_dir', 
                      # default='/content/drive/My Drive/d2d_forecasting/tests/')
                      default='/home/nidhisimmons/git/channel_prediction_2022/')

  args = parser.parse_args()

  if args.shift == None:
    args.shift = args.out_steps

  curr_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
  args.path_for_common_dir = args.path_for_common_dir + curr_time + '/'

  directory = args.path_for_common_dir
  if not os.path.isdir(directory):
    os.makedirs(directory)

  gpus = tf.config.experimental.list_physical_devices('GPU')
  if len(gpus) > 0:
    args.device = tf.config.list_physical_devices()[1]
  elif len(gpus) > 1:
    tf.config.experimental.set_visible_devices(devices=gpus[0:], device_type='GPU')
    tf.config.experimental.set_memory_growth(gpus[0:], True) 
    args.device = tf.config.list_physical_devices()[1:]
  else:
    args.device = tf.config.list_physical_devices()[0]

  return args


if __name__ == '__main__':
  args = parse_args()
  print('Called with args:')
  print(args)

  ######################################################################################################
  ############################################## DATA LOAD #############################################
  ######################################################################################################
  df = pd.read_csv(args.data_path, header=None, delimiter=r"\s+")
  if df.shape[0] == 1:
    df = df.T
  else:
    pass
  df = df.rename(columns={0: "fast-fading (dB)"})

  # For Holdout 70/20/10
  n = len(df)
  train_df = df[0:int(n*0.7)]
  val_df = df[int(n*0.7):int(n*0.9)]
  test_df = df[int(n*0.9):]

  # # For H samples predictions
  # n = len(df)
  # n2 = len(df[-s.OUT_STEPS-s.INPUT_WIDTH:])
  # train_df = df[0:int(n*0.7)]
  # val_df = df[int(n*0.7):int(n-n2)] 
  # test_df = df[-n2:]  

  train_mean = train_df.mean()
  train_std = train_df.std()
  train_min = train_df.min()
  train_max = train_df.max()
  # print('df shape', df.shape)
  num_features = df.shape[1]
  print('Number of features: ', num_features)
  print('train_min', train_min)
  print('train_max', train_max)

  ax = train_df.plot(figsize=(10,8))
  val_df.plot(ax=ax)
  test_df.plot(ax=ax)
  ax.legend(('Train', 'Val', 'Test'))
  plt.xlabel('Sample')
  plt.ylabel('Small-scale fading (dB)')
  plt.savefig(args.path_for_common_dir+'Dataset')

  if args.if_show_plots:
    plt.show()
  plt.close()
  

  ######################################################################################################
  ######################################## DATA NORMALIZATION ##########################################
  ######################################################################################################
  train_df, val_df, test_df = normalize_data(df, train_df, val_df, test_df, args.norm_type, args.if_show_plots)


  ######################################################################################################
  ########################################### BUILDING WINDOW ##########################################
  ######################################################################################################
  window = WindowGenerator(input_width=args.input_width,
                                 label_width=args.out_steps,
                                 shift=args.shift,
                                 train_df=train_df,
                                 val_df=val_df,
                                 test_df=test_df,
                                 train_min=train_min,
                                 train_max=train_max,
                                 batch_size=args.mb_size,
                                 inverse_function=min_max_inverse)

  if args.if_show_plots:
      window.plot()
      plt.suptitle('Multi window plot with inputs and labels')


  ######################################################################################################    
  #################################### FIT LINEAR REGRESSION ###########################################
  ######################################################################################################

  linear_models = linear_histories = [] # keras.sequential models and keras.history objects
  linear_mae = linear_rmse = linear_mae_rt = linear_rmse_rt = {} # errors dicts

  # fit model and output errors
  linear_models, linear_histories, \
  linear_mae, linear_rmse, linear_mae_rt, linear_rmse_rt = fitLinearRegression(args, window, train_min, train_max)

  ## plot and save (.txt) errors
  plot_errors(linear_mae, linear_rmse, linear_mae_rt, linear_rmse_rt, args.input_width, args.out_steps, args.num_units_l1, args.num_units_l2, args.path_for_common_dir, args.if_show_plots, model_name='LINEAR')
  pdb.set_trace()
  # Store best test error for each trial and model
  best_dense = []
  best_dense_ind = []
  best_lstm = []
  best_lstm_ind = []
  best_arlstm = []
  best_arlstm_ind = []
  best_gru = []
  best_gru_ind = []
  best_cnn = []
  best_cnn_ind = []
  best_nbeats = []
  best_nbeats_ind = []

  dense_mae_list = []
  lstm_mae_list = []
  arlstm_mae_list = []
  gru_mae_list = []
  cnn_mae_list = []
  nbeats_mae_list = []

  for i in range(args.n_trials):
    print('Trial: ', cont)

    # ######################################################################################################    
    # ############################################# FIT MLP ################################################
    # ######################################################################################################

    # dense_models = dense_histories = [] # keras.sequential models and keras.history objects
    # dense_mae = dense_rmse = dense_mae_rt = dense_rmse_rt = {} # errors dicts

    # # fit model and output errors
    # dense_models, dense_histories, \
    # dense_mae, dense_rmse, dense_mae_rt, dense_rmse_rt = fitMLP(args, window, train_min, train_max)

    # # Store best test MAE and index for a model
    # best_dense_ind.append(np.argmin(dense_mae['val']))
    # best_dense.append(np.min(dense_mae['val']))
    # dense_mae_list.append(dense_mae)

    # # plot and save (.txt) errors
    # plot_errors(dense_mae, dense_rmse, dense_mae_rt, dense_rmse_rt, args.input_width, args.out_steps, args.num_units_l1, args.num_units_l2, args.path_for_common_dir, args.if_show_plots, model_name='MLP')

    # ######################################################################################################    
    # ############################################# FIT LSTM ###############################################
    # ######################################################################################################

    # lstm_models = lstm_histories = [] # keras.sequential models and keras.history objects
    # lstm_mae = lstm_rmse = lstm_mae_rt = lstm_rmse_rt = {} # errors dicts

    # # fit model and output errors
    # lstm_models, lstm_histories, \
    # lstm_mae, lstm_rmse, lstm_mae_rt, lstm_rmse_rt = fitLSTM(args, window, train_min, train_max)

    # # Store best test MAE and index for a model
    # best_lstm_ind.append(np.argmin(lstm_mae['val']))
    # best_lstm.append(np.min(lstm_mae['val']))
    # lstm_mae_list.append(lstm_mae)

    # # plot and save (.txt) errors
    # plot_errors(lstm_mae, lstm_rmse, lstm_mae_rt, lstm_rmse_rt, args.input_width, args.out_steps, args.num_units_l1, args.num_units_l2, args.path_for_common_dir, args.if_show_plots, model_name='LSTM')

    # # ######################################################################################################    
    # # ########################################## FIT AR-LSTM ###############################################
    # # ######################################################################################################

    # # arlstm_models = arlstm_histories = [] # keras.sequential models and keras.history objects
    # # arlstm_mae = arlstm_rmse = arlstm_mae_rt = arlstm_rmse_rt = {} # errors dicts

    # # # fit model and output errors
    # # arlstm_models, arlstm_histories, \
    # # arlstm_mae, arlstm_rmse, arlstm_mae_rt, arlstm_rmse_rt = fitARLSTM(args, window, train_min, train_max)

    # # # Store best test MAE and index for a model
    # # best_arlstm_ind.append(np.argmin(arlstm_mae['val']))
    # # best_arlstm.append(np.min(arlstm_mae['val']))
    # # arlstm_mae_list.append(arlstm_mae)

    # # # plot and save (.txt) errors
    # # plot_errors(arlstm_mae, arlstm_rmse, arlstm_mae_rt, arlstm_rmse_rt, args.input_width, args.out_steps, args.num_units_l1, args.num_units_l2, args.path_for_common_dir, args.if_show_plots, model_name='AR-LSTM')

    # # ######################################################################################################    
    # # ############################################# FIT GRU ################################################
    # # ######################################################################################################

    # gru_models = gru_histories = [] # keras.sequential models and keras.history objects
    # gru_mae = gru_rmse = gru_mae_rt = gru_rmse_rt = {} # errors dicts

    # # fit model and output errors
    # gru_models, gru_histories, \
    # gru_mae, gru_rmse, gru_mae_rt, gru_rmse_rt = fitGRU(args, window, train_min, train_max)

    # # Store best test MAE and index for a model
    # best_gru_ind.append(np.argmin(gru_mae['val']))
    # best_gru.append(np.min(gru_mae['val']))
    # gru_mae_list.append(gru_mae)

    # # plot and save (.txt) errors
    # plot_errors(gru_mae, gru_rmse, gru_mae_rt, gru_rmse_rt, args.input_width, args.out_steps, args.num_units_l1, args.num_units_l2, args.path_for_common_dir, args.if_show_plots, model_name='GRU')

    # # ######################################################################################################    
    # # ############################################# FIT CNN ################################################
    # # ######################################################################################################

    # cnn_models = cnn_histories = [] # keras.sequential models and keras.history objects
    # cnn_mae = cnn_rmse = cnn_mae_rt = cnn_rmse_rt = {} # errors dicts

    # # fit model and output errors
    # cnn_models, cnn_histories, \
    # cnn_mae, cnn_rmse, cnn_mae_rt, cnn_rmse_rt = fitCNN(args, window, train_min, train_max)

    # # Store best test MAE and index for a model
    # best_cnn_ind.append(np.argmin(cnn_mae['val']))
    # best_cnn.append(np.min(cnn_mae['val']))
    # cnn_mae_list.append(cnn_mae)

    # # plot and save (.txt) errors
    # plot_errors(cnn_mae, cnn_rmse, cnn_mae_rt, cnn_rmse_rt, args.input_width, args.out_steps, args.num_units_l1, args.num_units_l2, args.path_for_common_dir, args.if_show_plots, model_name='1DCNN')

    #####################################################################################################    
    ######################################### FIT N-BEATS ###############################################
    #####################################################################################################

    nbeats_models = nbeats_histories = [] # keras.sequential models and keras.history objects
    nbeats_mae = nbeats_rmse = nbeats_mae_rt = nbeats_rmse_rt = {} # errors dicts

    # fit model and output errors
    nbeats_models, nbeats_histories, \
    nbeats_mae, nbeats_rmse, nbeats_mae_rt, nbeats_rmse_rt = fitNBEATS(args, window, train_min, train_max)

    # Store best test MAE and index for a model
    best_nbeats_ind.append(np.argmin(nbeats_mae['val']))
    best_nbeats.append(np.min(nbeats_mae['val']))
    nbeats_mae_list.append(nbeats_mae)

    # plot and save (.txt) errors
    plot_errors(nbeats_mae, nbeats_rmse, nbeats_mae_rt, nbeats_rmse_rt, args.input_width, args.out_steps, args.num_units_l1, args.num_units_l2, args.path_for_common_dir, args.if_show_plots, model_name='N-BEATS')

    cont +=1 # Trials counter update
    tf.keras.backend.clear_session()
    # end trials loop

  # tuple containing: best config index (1) and val MAE (2) for each trial, and val MAEs for all trials
  dense_tuple = (best_dense_ind, best_dense, dense_mae_list)
  lstm_tuple = (best_lstm_ind, best_lstm, lstm_mae_list)
  # arlstm_tuple = (best_arlstm_ind, best_arlstm, arlstm_mae_list)
  gru_tuple = (best_gru_ind, best_gru, gru_mae_list)
  cnn_tuple = (best_cnn_ind, best_cnn, cnn_mae_list)
  nbeats_tuple = (best_nbeats_ind, best_nbeats, nbeats_mae_list)

  # REPORT
  report_dict = {}

  print('#'*20+' LINEAR '+'#'*20)
  linear_result = linear_mae['val'][0]
  print(f'The mean val MAE is {linear_result}')
  report_dict['LINEAR'] = [np.nan, linear_result, np.nan, np.nan, np.nan]

  # report_dict['DENSE'] = get_report_dict(dense_tuple, 'DENSE', args.n_trials)
  # report_dict['LSTM'] = get_report_dict(lstm_tuple, 'LSTM', args.n_trials)
  # # report_dict['AR-LSTM'] = get_report_dict(arlstm_tuple, 'AR-LSTM', args.n_trials)
  # report_dict['GRU'] = get_report_dict(gru_tuple, 'GRU', args.n_trials)
  # report_dict['1DCNN'] = get_report_dict(cnn_tuple, '1DCNN', args.n_trials)
  report_dict['N-BEATS'] = get_report_dict(nbeats_tuple, 'N-BEATS', args.n_trials)

  report_df = pd.DataFrame(report_dict, columns=[x for x in report_dict.keys()], dtype='float32')
  report_df.rename(index={0:'Best Config', 1:'Mean MAE across trials', 2:'MAE std across trials', 3:'LB mean', 4:'LB std'}, inplace=True)

  print(report_df.transpose())

  with open(args.path_for_common_dir+'/results_report.txt', 'a') as f:
      f.write('- FINAL RESULTS -\n')
      f.write(f'args: {args}')
      dfAsString = report_df.transpose().to_string(header=True, index=True)
      f.write(dfAsString)

  # # For H samples test:
  # samples = test_df[:-12].values.T
  # result_nbeats = nbeats_models[best_nbeats_ind].predict(samples)
  # result_linear = linear_models[0].predict(samples)
  # result_dense = dense_models[best_dense_ind].predict(samples)
  # result_lstm = lstm_models[best_lstm_ind].predict(samples)
  # # result_arlstm = arlstm_models[best_arlstm_ind].predict(samples)
  # result_gru = gru_models[best_gru_ind].predict(samples)
  # result_cnn = cnn_models[best_cnn_ind].predict(samples)

  # plt.figure()
  # plt.title('Best Models Selected')
  # plt.plot(samples.T, label='Labels')
  # plt.plot(result_nbeats[0], label='N-BEATS')
  # plt.plot(result_linear[0], label='Linear')
  # plt.plot(result_dense[0], label='MLP')
  # plt.plot(result_lstm[0], label='LSTM')
  # # plt.plot(result_arlstm[0], label='AR-LSTM')
  # plt.plot(result_gru[0], label='GRU')
  # plt.plot(result_cnn[0], label='CNN')
  # plt.xlabel('Samples')
  # plt.ylabel('Small scale fading (dB) [norm]')
  # plt.legend()
  # plt.savefig(s.FIGURES_DIR+'TestResults')
  # plt.show()