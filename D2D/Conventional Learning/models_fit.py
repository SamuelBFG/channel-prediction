from aux_funcs import create_directory, save_parameters, compile_and_fit, get_errors
import matplotlib.pyplot as plt
import tensorflow as tf
from FeedBack import FeedBack
from NBEATSBlock import NBEATSBlock, NBEATS

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


