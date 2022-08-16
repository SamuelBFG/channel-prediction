import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import settings as s
import pandas as pd
import os
import pdb
from FeedBack import FeedBack
from NBEATSBlock import *
import argparse

def normalize_data(df, train_df, val_df, test_df, norm, show_plots):
  '''
  Function to normalize the data.
  Parameters:
      df: a univariate time series data, a dataframe with shape (X, 1)
      train_df: train part, a dataframe
      val_df: validation part, a dataframe
      test_df: test part, a dataframe
      norm: Normalization kind (0 for standardization, \
                                1 for centred mean and min = -1, \
                                2 for minmax), an integer in {0, 1, 2}
      show_plots: True to show the plots, a boolean
  Returns:
      train, validation, and test part of the data normalized, three dataframes
  '''

  train_mean = train_df.mean()
  train_std = train_df.std()
  train_min = train_df.min()
  train_max = train_df.max()

  print('train_min', train_min)
  print('train_max', train_max)

  #standardization
  if norm == 0: 
    
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

    plot_cols = 'Standardization'
    plot_features = df_std[plot_cols]
    _ = plot_features.plot()
    plt.ylabel('fast-fading (dB) [normed]')
    plt.xlabel('Sample number')

    if show_plots:
      plt.show()

  #centred mean and min value = -1 and max value < 1
  if norm == 1: 
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

    plot_cols = 'Normalized'
    plot_features = df_norm[plot_cols]
    _ = plot_features.plot()
    plt.ylabel('fast-fading (dB) [normed]')
    plt.xlabel('Sample number')
    
    if show_plots:
      plt.show()


  #minmax [-1,1] normaliztion
  if norm == 2: 
    # scaler = MinMaxScaler(feature_range=(-1,1))
    # scaled = scaler.fit_transform(train_df)

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

    plot_cols = 'MAX-MIN Normalized'
    plot_features = df_minmax[plot_cols]
    _ = plot_features.plot()
    plt.ylabel('fast-fading (dB) [normed]')
    plt.xlabel('Sample number')

    if show_plots:
      plt.show()

  return train_df, val_df, test_df

def min_max_inverse(data, train_min, train_max):
  '''
  Function to return the inverse min-max normalization.
  Parameters:
      data: windowed data, a dataframe
      train_min: minimum value of the training set, a scalar
      train_max: maximum value of the training set, a scalar
  Returns:
      inverse min-max normalization of data, a dataframe
  '''

  return train_min[0] + (train_max[0] - train_min[0]) * (data + 1)/2

def compile_and_fit(model, window, max_epochs, patience=10, file_name="models", save_weights_only=False):
  '''
  Function to compile and fit the keras model.
  Parameters:
      model: a keras model
      window: the sliding window, a WindowGenerator object
      max_epochs: the maximum epochs the model will be trained on, a scalar
      patience: number of epochs for patience for early stopping, a scalar
      file_name: the file name for saving purposes, a string
      save_weights_only: Truetf.keras.backend.clear_session() to save only the weights inside file_name, a boolean
  Returns:
      keras.History object containing training/validation losses for each epoch
  '''

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
  # pdb.set_trace()
  history = model.fit(window.train, epochs=max_epochs,
                      validation_data=window.val,
                      verbose = 0,
                      callbacks=[early_stopping, reduceLrOnPlateau])
  # print(history.history)  
  # pdb.set_trace()
  # if save_weights_only:
  #   model.save_weights(model_path)
  # else:
  #   model.save(file_name)
   
  # f = open(os.path.join(file_name,'losses.pckl'),'wb')
  # pickle.dump(history.history, f)
  # f.close()

  return history


def plot_errors(mae_dict, rmse_dict, maeRT_dict, rmseRT_dict, input_width, out_steps, num_units_l1, num_units_l2, figures_dir, show_plots, model_name):
  '''
  Function for plotting and saving the MAE, RMSE, MAE (reversescaled), and RMSE (reversescaled) errors.
  Parameters:
      mae_dict: MAE error (val/test) for each trained model, a dictionary
      rmse_dict: RMSE error (val/test) for each trained model, a dictionary
      maeRT_dict: MAE (reversescaled) error (val/test) for each trained model, a dictionary
      rmseRT_dict: RMSE (reversescaled) error (val/test) for each trained model, a dictionary
      input_width: window's input, a scalar
      out_steps: window's output, a scalar
      num_units_l1: configs (e.g., hidden units, filters) for model's layer 1, a list
      num_units_l2: configs (e.g., hidden units, filters) for model's layer 2, a list
      show_plots: True to show the plots, a boolean
      model_name: the model's name (e.g., LSTM, GRU), a string
  Returns:
      MAE, RMSE, MAE_RT, RMSE_RT plots for each given model as well as .txt files
  '''
  figures_dir = create_directory(model_name, input_width, out_steps, figures_dir)

  if model_name == '1DCNN':
    units = 'filters'
  elif model_name == 'AR-LSTM':
    num_units_l2 = []
    units = 'units'
  elif model_name == 'LINEAR':
    units = ''
    num_units_l1 = [0]
    num_units_l2 = []
  elif model_name == 'N-BEATS':
    units = 'units'
    num_units_l2 = []
  else:
    units = 'units'
  plt.figure()
  x = np.arange(len(rmse_dict['test']))
  width = 0.2

  plt.bar(x - 0.17, mae_dict['val'], width, label='Validation')
  plt.bar(x + 0.17, mae_dict['test'], width, label='Test')

  if len(num_units_l2) > 0:
    plt.xticks(ticks=x, labels = ['{x}-{y} '+str(units) for x,y in zip(num_units_l1, num_units_l2)],
            rotation=40)
  else:
    plt.xticks(ticks=x, labels = [str(x)+' '+str(units) for x in num_units_l1],
            rotation=45)

  plt.ylabel(f'Mean Absolute Error (MAE)')
  _ = plt.legend()
  plt.title('{} - Hyperparameters Search - Input: {} | Outuput: {}'.format(model_name, input_width, out_steps))
  plt.savefig(figures_dir+'/maes_'+str(model_name))

  if show_plots:
    plt.show()

  plt.close()
  plt.figure()
  plt.bar(x - 0.17, rmse_dict['val'], width, label='Validation')
  plt.bar(x + 0.17, rmse_dict['test'], width, label='Test')

  if len(num_units_l2) > 0:
    plt.xticks(ticks=x, labels = ['{x}-{y} '+str(units) for x,y in zip(num_units_l1, num_units_l2)],
            rotation=40)
  else:
    plt.xticks(ticks=x, labels = [str(x)+' '+str(units) for x in num_units_l1],
            rotation=45)

  plt.ylabel(f'Root Mean Square Error (RMSE)')
  _ = plt.legend()
  plt.title('{} - Hyperparameters Search - Input: {} | Outuput: {}'.format(model_name, input_width, out_steps))
  plt.savefig(figures_dir+'/rmses_'+str(model_name))
  if show_plots:
    plt.show()

  plt.close()
  plt.figure()
  plt.bar(x - 0.17, maeRT_dict['val'], width, label='Validation')
  plt.bar(x + 0.17, maeRT_dict['test'], width, label='Test')

  if len(num_units_l2) > 0:
    plt.xticks(ticks=x, labels = ['{x}-{y} '+str(units) for x,y in zip(num_units_l1, num_units_l2)],
            rotation=40)
  else:
    plt.xticks(ticks=x, labels = [str(x)+' '+str(units) for x in num_units_l1],
            rotation=45)

  plt.ylabel(f'Mean Absolute Error (MAE) - REVERSESCALED')
  _ = plt.legend()
  plt.title('{} - Hyperparameters Search - Input: {} | Outuput: {}'.format(model_name, input_width, out_steps))
  plt.savefig(figures_dir+'/maes_'+str(model_name)+'_RT')
  if show_plots:
    plt.show()

  plt.close()
  plt.figure()
  plt.bar(x - 0.17, rmseRT_dict['val'], width, label='Validation')
  plt.bar(x + 0.17, rmseRT_dict['test'], width, label='Test')

  if len(num_units_l2) > 0:
    plt.xticks(ticks=x, labels = ['{x}-{y} '+str(units) for x,y in zip(num_units_l1, num_units_l2)],
            rotation=40)
  else:
    plt.xticks(ticks=x, labels = [str(x)+' '+str(units) for x in num_units_l1],
            rotation=45)

  plt.ylabel(f'Root Mean Squared Error (RMSE) - REVERSESCALED')
  _ = plt.legend()
  plt.title('{} - Hyperparameters Search - Input: {} | Outuput: {}'.format(model_name, input_width, out_steps))
  plt.savefig(figures_dir+'/rmses_'+str(model_name)+'_RT')
  if show_plots:
    plt.show()
  plt.close()
  dt = {'MAE': [x for x in mae_dict['test']],
        'RMSE': [x for x in rmse_dict['test']]}

  if len(num_units_l2) > 0:
    df = pd.DataFrame(dt, index=['{x}-{y} '+str(units) for x,y in zip(num_units_l1, num_units_l2)])
  else:
    df = pd.DataFrame(dt, index=[str(x)+' '+str(units) for x in num_units_l1])

  print('{} - Errors Comparison'.format(model_name))
  print(df)

  with open(figures_dir+'/errors.txt', 'a') as f:
      f.write('{} - Errors Comparison\n'.format(model_name))
      dfAsString = df.to_string(header=True, index=True)
      f.write(dfAsString)

  dt = {'MAE': [x for x in maeRT_dict['test']],
        'RMSE': [x for x in rmseRT_dict['test']]}

  # df = pd.DataFrame(dt, index=[str(x) + ' units' for x in config])
  if len(num_units_l2) > 0:
    df = pd.DataFrame(dt, index=['{x}-{y} '+str(units) for x,y in zip(num_units_l1, num_units_l2)])
  else:
    df = pd.DataFrame(dt, index=[str(x)+' '+str(units) for x in num_units_l1])

  print('{} - Errors Comparison - REVERSESCALED'.format(model_name))
  print(df)

  with open(figures_dir+'/errors_reversescaled.txt', 'a') as f:
      f.write('{} - Errors Comparison - REVERSESCALED\n'.format(model_name))
      dfAsString = df.to_string(header=True, index=True)
      f.write(dfAsString)

def get_errors(model, window, model_name, train_min, train_max, batchsize):
  '''
  Function to get validation and test performance for a given model.
  Parameters:
      model: a keras Model
      window: the window for which the model was trained on, a WindowGenerator object
      model_name: the model's name (e.g., LSTM, GRU), a string
      train_min: minimum value of the training set, a scalar
      train_max: maximum value of the training set, a scalar
      batchsize: batchsize, a scalar
  Returns:
      validation and test performance (scaled and reversescaled) for a given model, four dictionaries
  '''

  val_performance = {}
  test_performance = {}
  val_performance_rt = {}
  test_performance_rt = {}

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
  for b in iter(window.val):

    #reverse transform the val_true and val_pred and then compute the metrics
    val_true = min_max_inverse(b[1], train_min, train_max) #val_true or val_labels
    val_pred = min_max_inverse(model.predict(b[0], verbose=0), train_min, train_max)
    
    diff_val = b[1] - model.predict(b[0], verbose=0) #val_labels (or val_true) - val_pred; scaled version
    diff_val_reversescaled = val_true - val_pred #val_labels (or val_true) - val_pred: reverse-scaled

    mae_val += np.mean(abs(diff_val))
    mse_val += np.mean(diff_val**2)

    mae_val_reversescaled += np.mean(abs(diff_val_reversescaled))
    mse_val_reversescaled += np.mean(diff_val_reversescaled**2)
  

  mae_val/=(len(window.val)-1)+(len(diff_val))/batchsize #(106+15/32)
  mse_val/=(len(window.val)-1)+(len(diff_val))/batchsize #(106+15/32)
  rmse_val = np.sqrt(mse_val)

  mae_val_reversescaled/=(len(window.val)-1)+(len(diff_val_reversescaled))/batchsize #(106+15/32)
  mse_val_reversescaled/=(len(window.val)-1)+(len(diff_val_reversescaled))/batchsize #(106+15/32)
  rmse_val_reversescaled = np.sqrt(mse_val_reversescaled)

  val_performance_rt[1] = mae_val_reversescaled
  val_performance_rt[2] = rmse_val_reversescaled

  val_performance[model_name] = model.evaluate(window.val, verbose=0)

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
  for c in iter(window.test):

    #reverse transform test_true and test_pred and then compute the metrics
    test_true = min_max_inverse(c[1], train_min, train_max) #test_labels
    test_pred = min_max_inverse(model.predict(c[0], verbose=0), train_min, train_max)
  
    diff_test = c[1] - model.predict(c[0], verbose=0) # test_labels (or test_true) - test_pred
    diff_test_reversescaled = test_true - test_pred #test_labels (or test_true) - test_pred: reverse-scaled

    mae_test += np.mean(abs(diff_test))
    mse_test += np.mean(diff_test**2)

    mae_test_reversescaled += np.mean(abs(diff_test_reversescaled))
    mse_test_reversescaled += np.mean(diff_test_reversescaled**2)

  mae_test /= (len(window.test)-1)+(len(diff_test))/batchsize 
  mse_test /= (len(window.test)-1)+(len(diff_test))/batchsize 
  rmse_test = np.sqrt(mse_test)

  mae_test_reversescaled/=(len(window.test)-1)+(len(diff_test_reversescaled))/batchsize 
  mse_test_reversescaled/=(len(window.test)-1)+(len(diff_test_reversescaled))/batchsize 
  rmse_test_reversescaled = np.sqrt(mse_test_reversescaled)

  test_performance_rt[1] = mae_test_reversescaled
  test_performance_rt[2] = rmse_test_reversescaled

  test_performance[model_name] = model.evaluate(window.test, verbose=0)


  return val_performance[model_name], test_performance[model_name], val_performance_rt, test_performance_rt

def create_directory(model_name, input_width, out_steps, figures_dir):
  '''
  Function to create a model subdirectory for saving purposes
  Parameters:
      model_name: the model's name (e.g., LSTM, GRU), a string
      input_width: window's input, a scalar
      out_steps: window's output, a scalar
      figures_dir: the main defined directory, a string
  Returns:
      a model subdirectory, a string
  '''
  directory = figures_dir+str(model_name)+'_input_'+str(input_width)+'_output_'+str(out_steps)
  if not os.path.isdir(directory):
    os.makedirs(directory)

  return directory

def save_parameters(model_name, figures_dir, input_width, out_steps, batchsize, num_units_l1, num_units_l2, 
  dropout_rate, max_epochs, conv_width=0, n_layers=0, n_blocks=0, b_sharing=False):
  '''
  Function to save all the hyperparameters used in a training.
  Parameters:
      model_name: the model's name (e.g., LSTM, GRU), a string
      figures_dir: the main defined directory, a string
      input_width: window's input, a scalar
      out_steps: window's output, a scalar
      num_units_l1: configs (e.g., hidden units, filters) for model's layer 1, a list
      num_units_l2: configs (e.g., hidden units, filters) for model's layer 2, a list
      dropout_rate: dropout rate, a scalar
      max_epochs: maximum epochs used to train the model, a scalar
      conv_width: the kernel size for conv models, a scalar
  Returns:
      saved .txt file containing the hyperparameters
  '''
  if model_name == 'LINEAR':
    df_params = {'input_width': [input_width],
                'output_steps': [out_steps],
                'batch size': [batchsize],
                'max_epochs': [max_epochs]}

  elif model_name == '1DCNN':
    df_params = {'input_width': [input_width],
                'output_steps': [out_steps],
                'num kernels L1': [num_units_l1],
                'num kernels L2': [num_units_l2],
                'kernel size': [conv_width],
                'dropout': [dropout_rate],
                'batch size': [batchsize],
                'max_epochs': [max_epochs]}

  elif model_name == 'AR-LSTM':
    df_params = {'input_width': [input_width],
                'output_steps': [out_steps],
                'num kernels L1': [num_units_l1],
                'batch size': [batchsize],
                'max_epochs': [max_epochs]}

  elif model_name == 'N-BEATS':
    df_params = {'input_width': [input_width],
                'output_steps': [out_steps],
                'block layers': [n_layers],
                'num blocks': [n_blocks],
                'num hidden units': [num_units_l1],
                'block sharing': [b_sharing],
                'batch size': [batchsize],
                'max_epochs': [max_epochs]}

  else:
    df_params = {'input_width': [input_width],
                'output_steps': [out_steps],
                'num hidden units L1': [num_units_l1],
                'num hidden units L2': [num_units_l2],
                'dropout': [dropout_rate],
                'batch size': [batchsize],
                'max_epochs': [max_epochs]}

  df_params = pd.DataFrame(df_params)
  print('{} - Hyperparameters'.format(model_name))
  print(df_params)

  with open(figures_dir+'/hyperparameters.txt', 'a') as f:
    f.write('{} - HYPERPARAMETERS\n'.format(model_name))
    dfAsString = df_params.to_string(header=True, index=True)
    f.write(dfAsString)

  return

def fitLinearRegression(args, window, train_min, train_max):
  '''
  Function to define and fit the baseline Linear Regression model.
  Parameters:
      window: the window for which the model was trained on, a WindowGenerator object
      train_min: minimum value of the training set, a scalar
      train_max: maximum value of the training set, a scalar
      input_width: window's input, a scalar
      out_steps: window's output, a scalar
      max_epochs: maximum epochs used to train the model, a scalar
      show_plots: True to show the plots, a boolean
      batchsize: number of samples in each mini batch, a integer
  Returns:
      saved figures regarding the window and the losses;
      models: a keras.Model object
      histories: a keras.History object
      mae: mae val/test errors, a dictionary
      rmse: rmse val/test errors, a dictionary
      mae_rt: mae (reversescaled) val/test errors, a dictionary
      rmse_rt: rmse (reversescaled) val/test errors, a dictionary

  '''
  model_name = 'LINEAR'
  figures_dir = create_directory(model_name, args.input_width, args.out_steps, args.path_for_common_dir)

  models = []
  histories = []

  val_mse_errors = []
  val_rmse_errors = [] 
  val_mae_errors = []
  val_mse_errors_reversescaled = []
  val_rmse_errors_reversescaled = [] 
  val_mae_errors_reversescaled = []

  test_mse_errors = []
  test_rmse_errors = [] 
  test_mae_errors = []
  test_mse_errors_reversescaled = []
  test_rmse_errors_reversescaled = [] 
  test_mae_errors_reversescaled = []

  # Storing the hyperparameters in a .txt file
  save_parameters(model_name, figures_dir, args.input_width, args.out_steps, args.mb_size, 0, 0, 0, args.max_epochs)

  print('##### {} MODEL'.format(model_name))

  model = tf.keras.Sequential([
      # Shape: (time, features) => (time*features)
      tf.keras.layers.Flatten(),
      # Shape => [batch, time*features, out_steps*features]
      tf.keras.layers.Dense(args.out_steps*window.get_num_features(),
                          kernel_initializer=tf.initializers.zeros()),
      # Shape => [batch, out_steps, features]
      tf.keras.layers.Reshape([args.out_steps, window.get_num_features()])
  ])

  models.append(model)
  history = compile_and_fit(model, window, args.max_epochs)
  histories.append(history)

  ## Calling get_errors function
  val_performance, test_performance, \
  val_performance_rt, test_performance_rt = get_errors(model, window, model_name, train_min, train_max, args.mb_size)

  # Storing MSE, MAE, RMSE val/test performance
  val_mse_errors.append(val_performance[0])
  val_mae_errors.append(val_performance[1]) 
  val_rmse_errors.append(val_performance[2])
  test_mse_errors.append(test_performance[0])
  test_mae_errors.append(test_performance[1]) 
  test_rmse_errors.append(test_performance[2])

  # Storing MSE, MAE, RMSE (reversescaled) val/test performance
  val_mae_errors_reversescaled.append(val_performance_rt[1])
  val_rmse_errors_reversescaled.append(val_performance_rt[2])
  test_mae_errors_reversescaled.append(test_performance_rt[1])
  test_rmse_errors_reversescaled.append(test_performance_rt[2])

  # if show_plots:
  window.plot(model)
  plt.suptitle('Model: {} | Input: {} | Output: {}'.format(model_name, args.input_width, args.out_steps))
  plt.savefig(figures_dir+'/window_'+str(model_name))
  if args.if_show_plots:
    plt.show()

  model_loss_train = history.history['loss']
  model_loss_val = history.history['val_loss']
  model_epochs =  range(len(model_loss_train))
  plt.close()
  plt.figure()
  plt.plot(model_epochs, model_loss_train, 'g', label='Training loss')
  plt.plot(model_epochs, model_loss_val, 'b', label='Validation loss')

  plt.title('{} Model Loss | Input: {} | Output: {}'.format(model_name, args.input_width, args.out_steps))
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.savefig(figures_dir+"/losses_"+str(model_name))

  if args.if_show_plots:
    plt.show()
  plt.close()


  mae = {'val':val_mae_errors,
          'test':test_mae_errors}
  
  rmse = {'val':val_rmse_errors,
          'test':test_rmse_errors}

  mae_rt = {'val':val_mae_errors_reversescaled,
            'test':test_mae_errors_reversescaled}

  rmse_rt = {'val':val_rmse_errors_reversescaled,
            'test':test_rmse_errors_reversescaled}

  return models, histories, mae, rmse, mae_rt, rmse_rt


def fitMLP(args, window, train_min, train_max):
  '''
  Function to define and fit a Multilayer Perceptron (MLP/DENSE/FFN) model.
  Parameters:
      window: the window for which the model was trained on, a WindowGenerator object
      train_min: minimum value of the training set, a scalar
      train_max: maximum value of the training set, a scalar
      input_width: window's input, a scalar
      out_steps: window's output, a scalar
      num_units_l1: configs (e.g., hidden units, filters) for model's layer 1, a list
      num_units_l2: configs (e.g., hidden units, filters) for model's layer 2, a list
      dropout_rate: dropout rate, a scalar
      max_epochs: maximum epochs used to train the model, a scalar
      show_plots: True to show the plots, a boolean
      batchsize: number of samples in each mini batch, a integer
  Returns:
      saved figures regarding the window and the losses;
      models: a list of keras.Model objects
      histories: a list of keras.History objects
      mae: mae val/test errors, a dictionary
      rmse: rmse val/test errors, a dictionary
      mae_rt: mae (reversescaled) val/test errors, a dictionary
      rmse_rt: rmse (reversescaled) val/test errors, a dictionary
  '''
  model_name = 'MLP'
  figures_dir = create_directory(model_name, args.input_width, args.out_steps, args.path_for_common_dir)

  models = []
  histories = []

  val_mse_errors = []
  val_rmse_errors = [] 
  val_mae_errors = []
  val_mse_errors_reversescaled = []
  val_rmse_errors_reversescaled = [] 
  val_mae_errors_reversescaled = []

  test_mse_errors = []
  test_rmse_errors = [] 
  test_mae_errors = []
  test_mse_errors_reversescaled = []
  test_rmse_errors_reversescaled = [] 
  test_mae_errors_reversescaled = []

  # Storing the hyperparameters in a .txt file
  save_parameters(model_name, figures_dir, args.input_width, args.out_steps, args.mb_size, args.num_units_l1, args.num_units_l2, args.dropout_rate, args.max_epochs)

  for i in range(len(args.num_units_l1)):
    if len(args.num_units_l2) > 0:
      print('##### {} MODEL WITH {} - {} UNITS'.format(model_name, args.num_units_l1[i], args.num_units_l2[i]))
      model = tf.keras.Sequential([
          # Shape [batch, time, features] => [batch, 1, features]
          # Shape [batch, time, features] => [batch, dense_units]
          # Adding more `dense_units` just overfits more quickly.
          # tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
          tf.keras.layers.Flatten(),
          # Shape => [batch, 1, dense_units]
          tf.keras.layers.Dense(args.num_units_l1[i], activation='relu'),
          # Adding dropout to prent overfitting
          tf.keras.layers.Dropout(args.dropout_rate),
          # second dense layer
          tf.keras.layers.Dense(args.num_units_l2[i], activation='relu'),
          # Shape => [batch, out_steps*features]
          # tf.keras.layers.Dense(out_steps*num_features),
          tf.keras.layers.Dense(args.out_steps*window.get_num_features()),
          # Shape => [batch, out_steps, features]
          tf.keras.layers.Reshape([args.out_steps, window.get_num_features()])
      ])
    else:
      print('##### {} MODEL WITH {} UNITS'.format(model_name, args.num_units_l1[i]))
      model = tf.keras.Sequential([                         
          # Shape [batch, time, features] => [batch, dense_units]
          # Adding more `dense_units` just overfits more quickly.
          tf.keras.layers.Flatten(),
          # Shape => [batch, 1, dense_units]
          tf.keras.layers.Dense(args.num_units_l1[i], activation='relu'),   
          # Shape => [batch, out_steps*features]
          tf.keras.layers.Dense(args.out_steps*window.get_num_features()),
          # Shape => [batch, out_steps, features]
          tf.keras.layers.Reshape([args.out_steps, window.get_num_features()])
        ])

    models.append(model)
    history = compile_and_fit(model, window, args.max_epochs)
    histories.append(history)

    ## Calling get_errors function
    val_performance, test_performance, \
    val_performance_rt, test_performance_rt = get_errors(model, window, model_name, train_min, train_max, args.mb_size)

    # Storing MSE, MAE, RMSE val/test performance
    val_mse_errors.append(val_performance[0])
    val_mae_errors.append(val_performance[1]) 
    val_rmse_errors.append(val_performance[2])
    test_mse_errors.append(test_performance[0])
    test_mae_errors.append(test_performance[1]) 
    test_rmse_errors.append(test_performance[2])

    # Storing MSE, MAE, RMSE (reversescaled) val/test performance
    val_mae_errors_reversescaled.append(val_performance_rt[1])
    val_rmse_errors_reversescaled.append(val_performance_rt[2])
    test_mae_errors_reversescaled.append(test_performance_rt[1])
    test_rmse_errors_reversescaled.append(test_performance_rt[2])

    
    # if show_plots:
    window.plot(model)
    if len(args.num_units_l2) > 0:
      plt.suptitle('Model: {} | Hidden Units: {} - {} | Input: {} | Output: {}'.format(model_name, args.num_units_l1[i], args.num_units_l2[i], args.input_width, args.out_steps))
      plt.savefig(figures_dir+'/window_'+str(model_name)+'_'+str(args.num_units_l1[i])+'_'+str(args.num_units_l2[i]))
    else:
      plt.suptitle('Model: {} | Hidden Units: {} | Input: {} | Output: {}'.format(model_name, args.num_units_l1[i], args.input_width, args.out_steps))
      plt.savefig(figures_dir+'/window_'+str(model_name)+'_'+str(args.num_units_l1[i]))
    if args.if_show_plots:
      plt.show()

    model_loss_train = history.history['loss']
    model_loss_val = history.history['val_loss']
    model_epochs =  range(len(model_loss_train))
    plt.close()
    plt.figure()
    plt.plot(model_epochs, model_loss_train, 'g', label='Training loss')
    plt.plot(model_epochs, model_loss_val, 'b', label='Validation loss')

    if len(args.num_units_l2) > 0:
      plt.title('{} Model Loss | Hidden Units: {} - {} | Input: {} | Output: {}'.format(model_name, args.num_units_l1[i], args.num_units_l2[i], args.input_width, args.out_steps))
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.legend()
      plt.savefig(figures_dir+"/losses_"+str(model_name)+'_'+str(args.num_units_l1[i])+'_'+str(args.num_units_l2[i]))
    else:
      plt.title('{} Model Loss | Hidden Units: {} | Input: {} | Output: {}'.format(model_name, args.num_units_l1[i], args.input_width, args.out_steps))
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.legend()
      plt.savefig(figures_dir+"/losses_"+str(model_name)+'_'+str(args.num_units_l1[i]))

    if args.if_show_plots:
      plt.show()
    plt.close()

  mae = {'val':val_mae_errors,
          'test':test_mae_errors}
  
  rmse = {'val':val_rmse_errors,
          'test':test_rmse_errors}

  mae_rt = {'val':val_mae_errors_reversescaled,
            'test':test_mae_errors_reversescaled}

  rmse_rt = {'val':val_rmse_errors_reversescaled,
            'test':test_rmse_errors_reversescaled}

  return models, histories, mae, rmse, mae_rt, rmse_rt


def fitLSTM(args, window, train_min, train_max):
  '''
  Function to define and fit a Long Short-term Memory (LSTM) model.
  Parameters:
      window: the window for which the model was trained on, a WindowGenerator object
      train_min: minimum value of the training set, a scalar
      train_max: maximum value of the training set, a scalar
      input_width: window's input, a scalar
      out_steps: window's output, a scalar
      num_units_l1: configs (e.g., hidden units, filters) for model's layer 1, a list
      num_units_l2: configs (e.g., hidden units, filters) for model's layer 2, a list
      dropout_rate: dropout rate, a scalar
      max_epochs: maximum epochs used to train the model, a scalar
      if_show_plots: True to show the plots, a boolean
      batchsize: number of samples in each mini batch, a integer
  Returns:
      saved figures regarding the window and the losses;
      models: a list of keras.Model objects
      histories: a list of keras.History objects
      mae: mae val/test errors, a dictionary
      rmse: rmse val/test errors, a dictionary
      mae_rt: mae (reversescaled) val/test errors, a dictionary
      rmse_rt: rmse (reversescaled) val/test errors, a dictionary
  '''
  model_name = 'LSTM'
  figures_dir = create_directory(model_name, args.input_width, args.out_steps, args.path_for_common_dir)

  models = []
  histories = []

  val_mse_errors = []
  val_rmse_errors = [] 
  val_mae_errors = []
  val_mse_errors_reversescaled = []
  val_rmse_errors_reversescaled = [] 
  val_mae_errors_reversescaled = []

  test_mse_errors = []
  test_rmse_errors = [] 
  test_mae_errors = []
  test_mse_errors_reversescaled = []
  test_rmse_errors_reversescaled = [] 
  test_mae_errors_reversescaled = []

  # Storing the hyperparameters in a .txt file
  save_parameters(model_name, figures_dir, args.input_width, args.out_steps, args.mb_size, args.num_units_l1, args.num_units_l2, args.dropout_rate, args.max_epochs)

  for i in range(len(args.num_units_l1)):
    if len(args.num_units_l2) > 0:
      print('##### {} MODEL WITH {} - {} UNITS'.format(model_name, args.num_units_l1[i], args.num_units_l2[i]))
      model = tf.keras.Sequential([
          # Shape [batch, time, features] => [batch, lstm_units]
          # Adding more `lstm_units` just overfits more quickly.
          tf.keras.layers.LSTM(args.num_units_l1[i], return_sequences=True),
          # Adding dropout to prent overfitting
          tf.keras.layers.Dropout(args.dropout_rate),
          # second lstm layer
          tf.keras.layers.LSTM(args.num_units_l2[i], return_sequences=False),
          # # # Adding dropout to prent overfitting
          tf.keras.layers.Dropout(args.dropout_rate),
          # Shape => [batch, out_steps*features]
          tf.keras.layers.Dense(args.out_steps*window.get_num_features()),
          # Shape => [batch, out_steps, features]
          tf.keras.layers.Reshape([args.out_steps, window.get_num_features()])
      ])
    else:
      print('##### {} MODEL WITH {} UNITS'.format(model_name, args.num_units_l1[i]))
      model = tf.keras.Sequential([                         
          # Shape [batch, time, features] => [batch, lstm_units]
          # Adding more `lstm_units` just overfits more quickly.
          tf.keras.layers.LSTM(args.num_units_l1[i], return_sequences=False),
          # Adding dropout to prent overfitting
          tf.keras.layers.Dropout(args.dropout_rate),
          # Shape => [batch, out_steps*features]
          tf.keras.layers.Dense(args.out_steps*window.get_num_features()),
          # Shape => [batch, aout_steps, features]
          tf.keras.layers.Reshape([args.out_steps, window.get_num_features()])
        ])
      
    models.append(model)
    history = compile_and_fit(model, window, args.max_epochs)
    histories.append(history)

    ## Calling get_errors function
    val_performance, test_performance, \
    val_performance_rt, test_performance_rt = get_errors(model, window, model_name, train_min, train_max, args.mb_size)

    # Storing MSE, MAE, RMSE val/test performance
    val_mse_errors.append(val_performance[0])
    val_mae_errors.append(val_performance[1]) 
    val_rmse_errors.append(val_performance[2])
    test_mse_errors.append(test_performance[0])
    test_mae_errors.append(test_performance[1]) 
    test_rmse_errors.append(test_performance[2])

    # Storing MSE, MAE, RMSE (reversescaled) val/test performance
    val_mae_errors_reversescaled.append(val_performance_rt[1])
    val_rmse_errors_reversescaled.append(val_performance_rt[2])
    test_mae_errors_reversescaled.append(test_performance_rt[1])
    test_rmse_errors_reversescaled.append(test_performance_rt[2])

    
  
    window.plot(model)
    if len(args.num_units_l2) > 0:
      plt.suptitle('Model: {} | Hidden Units: {} - {} | Input: {} | Output: {}'.format(model_name, args.num_units_l1[i], args.num_units_l2[i], args.input_width, args.out_steps))
      plt.savefig(figures_dir+'/window_'+str(model_name)+'_'+str(args.num_units_l1[i])+'_'+str(args.num_units_l2[i]))
    else:
      plt.suptitle('Model: {} | Hidden Units: {} | Input: {} | Output: {}'.format(model_name, args.num_units_l1[i], args.input_width, args.out_steps))
      plt.savefig(figures_dir+'/window_'+str(model_name)+'_'+str(args.num_units_l1[i]))
    if args.if_show_plots:
      plt.show()

    model_loss_train = history.history['loss']
    model_loss_val = history.history['val_loss']
    model_epochs =  range(len(model_loss_train))
    plt.close()
    plt.figure()
    plt.plot(model_epochs, model_loss_train, 'g', label='Training loss')
    plt.plot(model_epochs, model_loss_val, 'b', label='Validation loss')

    if len(args.num_units_l2) > 0:
      plt.title('{} Model Loss | Hidden Units: {} - {} | Input: {} | Output: {}'.format(model_name, args.num_units_l1[i], args.num_units_l2[i], args.input_width, args.out_steps))
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.legend()
      plt.savefig(figures_dir+"/losses_"+str(model_name)+'_'+str(args.num_units_l1[i])+'_'+str(args.num_units_l2[i]))
    else:
      plt.title('{} Model Loss | Hidden Units: {} | Input: {} | Output: {}'.format(model_name, args.num_units_l1[i], args.input_width, args.out_steps))
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.legend()
      plt.savefig(figures_dir+"/losses_"+str(model_name)+'_'+str(args.num_units_l1[i]))

    if args.if_show_plots:
      plt.show()
    plt.close()

  mae = {'val':val_mae_errors,
          'test':test_mae_errors}
  
  rmse = {'val':val_rmse_errors,
          'test':test_rmse_errors}

  mae_rt = {'val':val_mae_errors_reversescaled,
            'test':test_mae_errors_reversescaled}

  rmse_rt = {'val':val_rmse_errors_reversescaled,
            'test':test_rmse_errors_reversescaled}

  return models, histories, mae, rmse, mae_rt, rmse_rt

def fitARLSTM(args, window, train_min, train_max):
  '''
  Function to define and fit an Autoregressive Long Short-term Memory (AR-LSTM) model.
  Parameters:
      window: the window for which the model was trained on, a WindowGenerator object
      train_min: minimum value of the training set, a scalar
      train_max: maximum value of the training set, a scalar
      input_width: window's input, a scalar
      out_steps: window's output, a scalar
      num_units_l1: configs (e.g., hidden units, filters) for model's layer 1, a list
      dropout_rate: dropout rate, a scalar
      max_epochs: maximum epochs used to train the model, a scalar
      show_plots: True to show the plots, a boolean
      batchsize: number of samples in each mini batch, a integer
  Returns:
      saved figures regarding the window and the losses;
      models: a list of keras.Model objects
      histories: a list of keras.History objects
      mae: mae val/test errors, a dictionary
      rmse: rmse val/test errors, a dictionary
      mae_rt: mae (reversescaled) val/test errors, a dictionary
      rmse_rt: rmse (reversescaled) val/test errors, a dictionary
  '''
  model_name = 'AR-LSTM'
  figures_dir = create_directory(model_name, args.input_width, args.out_steps, args.path_for_common_dir)

  models = []
  histories = []

  val_mse_errors = []
  val_rmse_errors = [] 
  val_mae_errors = []
  val_mse_errors_reversescaled = []
  val_rmse_errors_reversescaled = [] 
  val_mae_errors_reversescaled = []

  test_mse_errors = []
  test_rmse_errors = [] 
  test_mae_errors = []
  test_mse_errors_reversescaled = []
  test_rmse_errors_reversescaled = [] 
  test_mae_errors_reversescaled = []

  # Storing the hyperparameters in a .txt file
  save_parameters(model_name, figures_dir, args.input_width, args.out_steps, args.mb_size, args.num_units_l1, 0, 0, args.max_epochs)

  for i in range(len(args.num_units_l1)):
    print('##### {} MODEL WITH {} UNITS'.format(model_name, args.num_units_l1[i]))
    model = FeedBack(units=args.num_units_l1[i], out_steps=args.out_steps)
    models.append(model)
    history = compile_and_fit(model, window, args.max_epochs)
    histories.append(history)

    ## Calling get_errors function
    val_performance, test_performance, \
    val_performance_rt, test_performance_rt = get_errors(model, window, model_name, train_min, train_max, args.mb_size)

    # Storing MSE, MAE, RMSE val/test performance
    val_mse_errors.append(val_performance[0])
    val_mae_errors.append(val_performance[1]) 
    val_rmse_errors.append(val_performance[2])
    test_mse_errors.append(test_performance[0])
    test_mae_errors.append(test_performance[1]) 
    test_rmse_errors.append(test_performance[2])

    # Storing MSE, MAE, RMSE (reversescaled) val/test performance
    val_mae_errors_reversescaled.append(val_performance_rt[1])
    val_rmse_errors_reversescaled.append(val_performance_rt[2])
    test_mae_errors_reversescaled.append(test_performance_rt[1])
    test_rmse_errors_reversescaled.append(test_performance_rt[2])

    window.plot(model)
    plt.suptitle('Model: {} | Hidden Units: {} | Input: {} | Output: {}'.format(model_name, args.num_units_l1[i], args.input_width, args.out_steps))
    plt.savefig(figures_dir+'/window_'+str(model_name)+'_'+str(args.num_units_l1[i]))
    if args.if_show_plots:
      plt.show()

    model_loss_train = history.history['loss']
    model_loss_val = history.history['val_loss']
    model_epochs =  range(len(model_loss_train))
    plt.close()
    plt.figure()
    plt.plot(model_epochs, model_loss_train, 'g', label='Training loss')
    plt.plot(model_epochs, model_loss_val, 'b', label='Validation loss')


    plt.title('{} Model Loss | Hidden Units: {} | Input: {} | Output: {}'.format(model_name, args.num_units_l1[i], args.input_width, args.out_steps))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(figures_dir+"/losses_"+str(model_name)+'_'+str(args.num_units_l1[i]))

    if args.if_show_plots:
      plt.show()
    plt.close()

  mae = {'val':val_mae_errors,
          'test':test_mae_errors}
  
  rmse = {'val':val_rmse_errors,
          'test':test_rmse_errors}

  mae_rt = {'val':val_mae_errors_reversescaled,
            'test':test_mae_errors_reversescaled}

  rmse_rt = {'val':val_rmse_errors_reversescaled,
            'test':test_rmse_errors_reversescaled}

  return models, histories, mae, rmse, mae_rt, rmse_rt

def fitGRU(args, window, train_min, train_max):
  '''
  Function to define and fit a Gated Recurrent Unit (GRU) model.
  Parameters:
      window: the window for which the model was trained on, a WindowGenerator object
      train_min: minimum value of the training set, a scalar
      train_max: maximum value of the training set, a scalar
      input_width: window's input, a scalar
      out_steps: window's output, a scalar
      num_units_l1: configs (e.g., hidden units, filters) for model's layer 1, a list
      num_units_l2: configs (e.g., hidden units, filters) for model's layer 2, a list
      dropout_rate: dropout rate, a scalar
      max_epochs: maximum epochs used to train the model, a scalar
      show_plots: True to show the plots, a boolean
      batchsize: number of samples in each mini batch, a integer
  Returns:
      saved figures regarding the window and the losses;
      models: a list of keras.Model objects
      histories: a list of keras.History objects
      mae: mae val/test errors, a dictionary
      rmse: rmse val/test errors, a dictionary
      mae_rt: mae (reversescaled) val/test errors, a dictionary
      rmse_rt: rmse (reversescaled) val/test errors, a dictionary
  '''
  model_name = 'GRU'
  figures_dir = create_directory(model_name, args.input_width, args.out_steps, args.path_for_common_dir)
  models = []
  histories = []

  val_mse_errors = []
  val_rmse_errors = [] 
  val_mae_errors = []
  val_mse_errors_reversescaled = []
  val_rmse_errors_reversescaled = [] 
  val_mae_errors_reversescaled = []

  test_mse_errors = []
  test_rmse_errors = [] 
  test_mae_errors = []
  test_mse_errors_reversescaled = []
  test_rmse_errors_reversescaled = [] 
  test_mae_errors_reversescaled = []

  # Storing the hyperparameters in a .txt file
  save_parameters(model_name, figures_dir, args.input_width, args.out_steps, args.mb_size, args.num_units_l1, args.num_units_l2, args.dropout_rate, args.max_epochs)

  for i in range(len(args.num_units_l1)):
    if len(args.num_units_l2) > 0:
      print('##### {} MODEL WITH {} - {} UNITS'.format(model_name, args.num_units_l1[i], args.num_units_l2[i]))
      model = tf.keras.Sequential([
          # Shape [batch, time, features] => [batch, gru_units]
          # Adding more `gru_units` just overfits more quickly.
          tf.keras.layers.GRU(args.num_units_l1[i], return_sequences=True),
          # Adding dropout to prent overfitting
          tf.keras.layers.Dropout(args.dropout_rate),
          # second gru layer
          tf.keras.layers.GRU(args.num_units_l2[i], return_sequences=False),
          # # # Adding dropout to prent overfitting
          tf.keras.layers.Dropout(args.dropout_rate),
          # Shape => [batch, out_steps*features]
          tf.keras.layers.Dense(args.out_steps*window.get_num_features()),
          # Shape => [batch, out_steps, features]
          tf.keras.layers.Reshape([args.out_steps, window.get_num_features()])
      ])
    else:
      print('##### {} MODEL WITH {} UNITS'.format(model_name, args.num_units_l1[i]))
      model = tf.keras.Sequential([                         
          # Shape [batch, time, features] => [batch, gru_units]
          # Adding more `gru_units` just overfits more quickly.
          tf.keras.layers.GRU(args.num_units_l1[i], return_sequences=False),
          # Adding dropout to prent overfitting
          tf.keras.layers.Dropout(args.dropout_rate),
          # Shape => [batch, out_steps*features]
          tf.keras.layers.Dense(args.out_steps*window.get_num_features()),
          # Shape => [batch, out_steps, features]
          tf.keras.layers.Reshape([args.out_steps, window.get_num_features()])
        ])
        
    models.append(model)
    history = compile_and_fit(model, window, args.max_epochs)
    histories.append(history)

    ## Calling get_errors function
    val_performance, test_performance, \
    val_performance_rt, test_performance_rt = get_errors(model, window, model_name, train_min, train_max, args.mb_size)

    # Storing MSE, MAE, RMSE val/test performance
    val_mse_errors.append(val_performance[0])
    val_mae_errors.append(val_performance[1]) 
    val_rmse_errors.append(val_performance[2])
    test_mse_errors.append(test_performance[0])
    test_mae_errors.append(test_performance[1]) 
    test_rmse_errors.append(test_performance[2])

    # Storing MSE, MAE, RMSE (reversescaled) val/test performance
    val_mae_errors_reversescaled.append(val_performance_rt[1])
    val_rmse_errors_reversescaled.append(val_performance_rt[2])
    test_mae_errors_reversescaled.append(test_performance_rt[1])
    test_rmse_errors_reversescaled.append(test_performance_rt[2])

    
    # if show_plots:
    window.plot(model)
    if len(args.num_units_l2) > 0:
      plt.suptitle('Model: {} | Hidden Units: {} - {} | Input: {} | Output: {}'.format(model_name, args.num_units_l1[i], args.num_units_l2[i], args.input_width, args.out_steps))
      plt.savefig(figures_dir+'/window_'+str(model_name)+'_'+str(args.num_units_l1[i])+'_'+str(args.num_units_l2[i]))
    else:
      plt.suptitle('Model: {} | Hidden Units: {} | Input: {} | Output: {}'.format(model_name, args.num_units_l1[i], args.input_width, args.out_steps))
      plt.savefig(figures_dir+'/window_'+str(model_name)+'_'+str(args.num_units_l1[i]))
    if args.if_show_plots:
      plt.show()

    model_loss_train = history.history['loss']
    model_loss_val = history.history['val_loss']
    model_epochs =  range(len(model_loss_train))
    plt.close()
    plt.figure()
    plt.plot(model_epochs, model_loss_train, 'g', label='Training loss')
    plt.plot(model_epochs, model_loss_val, 'b', label='Validation loss')

    if len(args.num_units_l2) > 0:
      plt.title('{} Model Loss | Hidden Units: {} - {} | Input: {} | Output: {}'.format(model_name, args.num_units_l1[i], args.num_units_l2[i], args.input_width, args.out_steps))
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.legend()
      plt.savefig(figures_dir+"/losses_"+str(model_name)+'_'+str(args.num_units_l1[i])+'_'+str(args.num_units_l2[i]))
    else:
      plt.title('{} Model Loss | Hidden Units: {} | Input: {} | Output: {}'.format(model_name, args.num_units_l1[i], args.input_width, args.out_steps))
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.legend()
      plt.savefig(figures_dir+"/losses_"+str(model_name)+'_'+str(args.num_units_l1[i]))

    if args.if_show_plots:
      plt.show()
    plt.close()

  mae = {'val':val_mae_errors,
          'test':test_mae_errors}
  
  rmse = {'val':val_rmse_errors,
          'test':test_rmse_errors}

  mae_rt = {'val':val_mae_errors_reversescaled,
            'test':test_mae_errors_reversescaled}

  rmse_rt = {'val':val_rmse_errors_reversescaled,
            'test':test_rmse_errors_reversescaled}

  return models, histories, mae, rmse, mae_rt, rmse_rt


# def fitCNN(window, train_min, train_max, input_width, out_steps, conv_width, num_units_l1, num_units_l2, dropout_rate, max_epochs, show_plots, batchsize):
def fitCNN(args, window, train_min, train_max):
  '''
  Function to define and fit a 1D Convolutional Neural Network (1DCNN) model.
  Parameters:
      window: the window for which the model was trained on, a WindowGenerator object
      train_min: minimum value of the training set, a scalar
      train_max: maximum value of the training set, a scalar
      input_width: window's input, a scalar
      out_steps: window's output, a scalar
      conv_width: kernel size, a scalar
      num_units_l1: configs (e.g., hidden units, filters) for model's layer 1, a list
      num_units_l2: configs (e.g., hidden units, filters) for model's layer 2, a list
      dropout_rate: dropout rate, a scalar
      max_epochs: maximum epochs used to train the model, a scalar
      show_plots: True to show the plots, a boolean
      batchsize: number of samples in each mini batch, a integer
  Returns:
      saved figures regarding the window and the losses;
      models: a list of keras.Model objects
      histories: a list of keras.History objects
      mae: mae val/test errors, a dictionary
      rmse: rmse val/test errors, a dictionary
      mae_rt: mae (reversescaled) val/test errors, a dictionary
      rmse_rt: rmse (reversescaled) val/test errors, a dictionary
  '''
  model_name = '1DCNN'
  figures_dir = create_directory(model_name, args.input_width, args.out_steps, args.path_for_common_dir)

  models = []
  histories = []

  val_mse_errors = []
  val_rmse_errors = [] 
  val_mae_errors = []
  val_mse_errors_reversescaled = []
  val_rmse_errors_reversescaled = [] 
  val_mae_errors_reversescaled = []

  test_mse_errors = []
  test_rmse_errors = [] 
  test_mae_errors = []
  test_mse_errors_reversescaled = []
  test_rmse_errors_reversescaled = [] 
  test_mae_errors_reversescaled = []

  # Storing the hyperparameters in a .txt file
  save_parameters(model_name, figures_dir, args.input_width, args.out_steps, args.mb_size, args.num_units_l1, args.num_units_l2, args.dropout_rate, args.max_epochs, args.conv_width)

  for i in range(len(args.num_units_l1)):
    if len(args.num_units_l2) > 0:
      print('##### {} MODEL WITH {} - {} FILTERS | KERNEL SIZE: {}'.format(model_name, args.num_units_l1[i], args.num_units_l2[i], args.conv_width))
      model = tf.keras.Sequential([
          # Shape [batch, time, features] => [batch, conv_width, features]  
          tf.keras.layers.Lambda(lambda x: x[:, -args.conv_width:, :]),
          ######      Shape [batch, time, features] => [batch, INPUT_SIZE, features]
          tf.keras.layers.Conv1D(filters=args.num_units_l1[i],
                                 activation='relu',
                                 kernel_size=(args.conv_width),
                                 padding='same',
                                 strides=1),
          # Adding dropout to prent overfitting
          tf.keras.layers.Dropout(args.dropout_rate),
          # second convolutional layer
          tf.keras.layers.Conv1D(filters=args.num_units_l2[i],
                                 activation='relu',
                                 kernel_size=(args.conv_width),
                                 padding='same',
                                 strides=1),
          # # # Adding dropout to prent overfitting
          tf.keras.layers.Dropout(args.dropout_rate),
          # Flatten
          tf.keras.layers.Flatten(),
          # Shape => [batch, out_steps*features]
          tf.keras.layers.Dense(args.out_steps*window.get_num_features()),
          # Shape => [batch, out_steps, features]
          tf.keras.layers.Reshape([args.out_steps, window.get_num_features()])
      ])
    else:
      print('##### {} MODEL WITH {} FILTERS | KERNEL SIZE: {}'.format(model_name, args.num_units_l1[i], args.conv_width))
      model = tf.keras.Sequential([                         
          # Shape [batch, time, features] => [batch, conv_width, features]  
          tf.keras.layers.Lambda(lambda x: x[:, -args.conv_width:, :]),
          tf.keras.layers.Conv1D(filters=args.num_units_l1[i],
                                 activation='relu',
                                 kernel_size=(args.conv_width),
                                 padding='same',
                                 strides=1),
          # Adding dropout to prent overfitting
          tf.keras.layers.Dropout(args.dropout_rate),
          # Flatten
          tf.keras.layers.Flatten(),
          # Shape => [batch, out_steps*features]
          tf.keras.layers.Dense(args.out_steps*window.get_num_features()),
          # Shape => [batch, out_steps, features]
          tf.keras.layers.Reshape([args.out_steps, window.get_num_features()])
        ])

    models.append(model)
    history = compile_and_fit(model, window, args.max_epochs)
    histories.append(history)

    ## Calling get_errors function
    val_performance, test_performance, \
    val_performance_rt, test_performance_rt = get_errors(model, window, model_name, train_min, train_max, args.mb_size)

    # Storing MSE, MAE, RMSE val/test performance
    val_mse_errors.append(val_performance[0])
    val_mae_errors.append(val_performance[1]) 
    val_rmse_errors.append(val_performance[2])
    test_mse_errors.append(test_performance[0])
    test_mae_errors.append(test_performance[1]) 
    test_rmse_errors.append(test_performance[2])

    # Storing MSE, MAE, RMSE (reversescaled) val/test performance
    val_mae_errors_reversescaled.append(val_performance_rt[1])
    val_rmse_errors_reversescaled.append(val_performance_rt[2])
    test_mae_errors_reversescaled.append(test_performance_rt[1])
    test_rmse_errors_reversescaled.append(test_performance_rt[2])

    
    # if show_plots:
    window.plot(model)
    if len(args.num_units_l2) > 0:
      plt.suptitle('Model: {} | # Filters: {} - {} | Kernel Size: {} | Input: {} | Output: {}'.format(model_name, args.num_units_l1[i], args.num_units_l2[i], conv_width, input_width, out_steps))
      plt.savefig(figures_dir+'/window_'+str(model_name)+'_'+str(args.num_units_l1[i])+'_'+str(args.num_units_l2[i]))
    else:
      plt.suptitle('Model: {} | # Filters: {} | Kernel Size: {} | Input: {} | Output: {}'.format(model_name, args.num_units_l1[i], args.conv_width, args.input_width, args.out_steps))
      plt.savefig(figures_dir+'/window_'+str(model_name)+'_'+str(args.num_units_l1[i]))
    if args.if_show_plots:
      plt.show()

    model_loss_train = history.history['loss']
    model_loss_val = history.history['val_loss']
    model_epochs =  range(len(model_loss_train))
    plt.close()
    plt.figure()
    plt.plot(model_epochs, model_loss_train, 'g', label='Training loss')
    plt.plot(model_epochs, model_loss_val, 'b', label='Validation loss')

    if len(args.num_units_l2) > 0:
      plt.title('{} Model Loss | # Filters: {} - {} | Kernel Size: {} | Input: {} | Output: {}'.format(model_name, args.num_units_l1[i], args.num_units_l2[i], args.conv_width, args.input_width, args.out_steps))
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.legend()
      plt.savefig(figures_dir+"/losses_"+str(model_name)+'_'+str(args.num_units_l1[i])+'_'+str(args.num_units_l2[i]))
    else:
      plt.title('{} Model Loss | # Filters: {} | Kernel Size: {} | Input: {} | Output: {}'.format(model_name, args.num_units_l1[i], args.conv_width, args.input_width, args.out_steps))
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.legend()
      plt.savefig(figures_dir+"/losses_"+str(model_name)+'_'+str(args.num_units_l1[i]))

    if args.if_show_plots:
      plt.show()
    plt.close()

  mae = {'val':val_mae_errors,
          'test':test_mae_errors}
  
  rmse = {'val':val_rmse_errors,
          'test':test_rmse_errors}

  mae_rt = {'val':val_mae_errors_reversescaled,
            'test':test_mae_errors_reversescaled}

  rmse_rt = {'val':val_rmse_errors_reversescaled,
            'test':test_rmse_errors_reversescaled}

  return models, histories, mae, rmse, mae_rt, rmse_rt


# def fitNBEATS(window, train_min, train_max, input_width, out_steps, num_units_l1, n_blocks, n_layers, b_sharing, max_epochs, show_plots, batchsize):
def fitNBEATS(args, window, train_min, train_max):
  '''
  Function to define and fit a generic architecture of N-BEATS model.
  Parameters:
      window: the window for which the model was trained on, a WindowGenerator object
      train_min: minimum value of the training set, a scalar
      train_max: maximum value of the training set, a scalar
      input_width: window's input, a scalar
      out_steps: window's output, a scalar
      num_units_l1: configs (e.g., hidden units, filters) for model's block, a list
      n_blocks: number of blocks for the generic architecture, a integer
      n_layers: number of layers for each fully-connected network, a integer
      b_sharing: True to share weights between the blocks, a boolean
      max_epochs: maximum epochs used to train the model, a scalar
      show_plots: True to show the plots, a boolean
      mb_size: number of samples in each mini batch, a integer
  Returns:
      saved figures regarding the window and the losses;
      models: a list of keras.Model objects
      histories: a list of keras.History objects
      mae: mae val/test errors, a dictionary
      rmse: rmse val/test errors, a dictionary
      mae_rt: mae (reversescaled) val/test errors, a dictionary
      rmse_rt: rmse (reversescaled) val/test errors, a dictionary
  '''
  model_name = 'N-BEATS'
  figures_dir = create_directory(model_name, args.input_width, args.out_steps, args.path_for_common_dir)

  models = []
  histories = []

  val_mse_errors = []
  val_rmse_errors = [] 
  val_mae_errors = []
  val_mse_errors_reversescaled = []
  val_rmse_errors_reversescaled = [] 
  val_mae_errors_reversescaled = []

  test_mse_errors = []
  test_rmse_errors = [] 
  test_mae_errors = []
  test_mse_errors_reversescaled = []
  test_rmse_errors_reversescaled = [] 
  test_mae_errors_reversescaled = []
           
  # Storing the hyperparameters in a .txt file
  save_parameters(model_name, figures_dir, args.input_width, args.out_steps, args.mb_size, args.num_units_l1, 0, 0, args.max_epochs, 0, args.n_layers, args.n_blocks, args.b_sharing)

  for i in range(len(args.num_units_l1)):
    print('##### {} MODEL WITH {} HIDDEN UNITS'.format(model_name, args.num_units_l1[i]))
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(args.input_width,)))
    nbeats = NBEATS(input_size=args.input_width, output_size=args.out_steps, block_layers=args.n_layers, num_blocks=args.n_blocks, hidden_units=args.num_units_l1[i], block_sharing=args.b_sharing)
    model.add(nbeats)
    model.add(tf.keras.layers.Reshape([args.out_steps, window.get_num_features()]))
    
    models.append(model)
    history = compile_and_fit(model, window, args.max_epochs)
    histories.append(history)

    ## Calling get_errors function
    val_performance, test_performance, \
    val_performance_rt, test_performance_rt = get_errors(model, window, model_name, train_min, train_max, args.mb_size)

    # Storing MSE, MAE, RMSE val/test performance
    val_mse_errors.append(val_performance[0])
    val_mae_errors.append(val_performance[1]) 
    val_rmse_errors.append(val_performance[2])
    test_mse_errors.append(test_performance[0])
    test_mae_errors.append(test_performance[1]) 
    test_rmse_errors.append(test_performance[2])

    # Storing MSE, MAE, RMSE (reversescaled) val/test performance
    val_mae_errors_reversescaled.append(val_performance_rt[1])
    val_rmse_errors_reversescaled.append(val_performance_rt[2])
    test_mae_errors_reversescaled.append(test_performance_rt[1])
    test_rmse_errors_reversescaled.append(test_performance_rt[2])

    # if show_plots:
    window.plot(model)
    plt.suptitle('Model: {} | # Hidden Units: {} | # Layers: {} | # Blocks: {} | Block Sharing? {} | Input: {} | Output: {}'.format(model_name, args.num_units_l1[i], args.n_layers, args.n_blocks, args.b_sharing, args.input_width, args.out_steps))
    plt.savefig(figures_dir+'/window_'+str(model_name)+'_'+str(args.num_units_l1[i]))
    if args.if_show_plots:
      plt.show()

    model_loss_train = history.history['loss']
    model_loss_val = history.history['val_loss']
    model_epochs =  range(len(model_loss_train))
    plt.close()
    plt.figure()
    plt.plot(model_epochs, model_loss_train, 'g', label='Training loss')
    plt.plot(model_epochs, model_loss_val, 'b', label='Validation loss')

    plt.title('{} Model Loss | # Hidden Units: {} | # Layers: {} | # Blocks: {} | Block Sharing? {} | Input: {} | Output: {}'.format(model_name, args.num_units_l1[i], args.n_layers, args.n_blocks, args.b_sharing, args.input_width, args.out_steps))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(figures_dir+"/losses_"+str(model_name)+'_'+str(args.num_units_l1[i]))

    if args.if_show_plots:
      plt.show()
    plt.close()

    # tf.keras.backend.clear_session()

  mae = {'val':val_mae_errors,
          'test':test_mae_errors}
  
  rmse = {'val':val_rmse_errors,
          'test':test_rmse_errors}

  mae_rt = {'val':val_mae_errors_reversescaled,
            'test':test_mae_errors_reversescaled}

  rmse_rt = {'val':val_rmse_errors_reversescaled,
            'test':test_rmse_errors_reversescaled}

  return models, histories, mae, rmse, mae_rt, rmse_rt


