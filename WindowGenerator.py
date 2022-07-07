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
from tensorflow import keras
import pickle
import warnings
from tensorflow import keras
from keras.callbacks import History
from sklearn.preprocessing import MinMaxScaler
import settings as s

# ## For ARIMA
# import statsmodels.api as sm
# from statsmodels.graphics.tsaplots import plot_acf
# from statsmodels.graphics.tsaplots import plot_pacf
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.tsa.ar_model import AutoReg
# from statsmodels.tsa.arima.model import ARIMA
# import pmdarima
# from pmdarima import auto_arima
# from statsmodels.tsa.statespace.sarimax import SARIMAX

# ##

# ## For Prophet
# from fbprophet import Prophet
## 


# %% [markdown]
# ## Data windowing
# The models in this tutorial will make a set of predictions based on a window of consecutive samples from the data. 
# The main features of the input windows are:
# * The width (number of time steps) of the input and label windows
# * The time offset between them.
# * Which features are used as inputs, labels, or both. 
# This section focuses on implementing the data windowing so that it can be reused for all of the models.
 
# %% [markdown]
# Depending on the task and type of model you may want to generate a variety of data windows. Here are some examples:
# 1. For example, to make a single prediction 24h into the future, given 24h of history you might define a window like this:
#   Input width = 24; offset = 24; Label width = 1; Total width = 48
# 2. A model that makes a prediction 1h into the future, given 6h of history would need a window like this:
#   Input width = 6; offset = 1; Label width = 1; Total width = 7
# %% [markdown]
# The rest of this section defines a `WindowGenerator` class. This class can:
# 1. Handle the indexes and offsets as shown above.
# 2. Split windows of features into a `(features, labels)` pairs.
# 3. Plot the content of the resulting windows.
# 4. Efficiently generate batches of these windows from the training, evaluation, and test data, using `tf.data.Dataset`s.
# %% [markdown]
# ### 1. Indexes and offsets
# Start by creating the `WindowGenerator` class. The `__init__` method includes all the necessary 
# logic for the input and label indices.
# It also takes the train, eval, and test dataframes as input. 
# These will be converted to `tf.data.Dataset`s of windows later.

# %%
class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df, val_df, test_df, train_min, train_max,
               batch_size, label_columns=None, inverse_function=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df
    self.inverse_function=inverse_function
    self.train_min = train_min
    self.train_max = train_max
    self.batch_size = batch_size
    
    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

  def get_num_features(self):
    return self.train_df.shape[1]
    
# %%
def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window


# %% [markdown]
# Typically data in TensorFlow is packed into arrays where the outermost index is across examples (the "batch" dimension). 
# The middle indices are the "time" or "space" (width, height) dimension(s). The innermost indices are the features.
# The code above took a batch of 3, 7-timestep windows, with 1 feature at each time step. 
# It split them into a batch of 6-timestep, 1 feature input, and a 1-timestep 1-feature label. 
# The label only has one feature because the `WindowGenerator` was initialized with `label_columns=['P (dBm)']`. 

# %%
def plot(self, model=None, plot_col='fast-fading (dB)', max_subplots=3):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    #print(n)
    plt.subplot(3, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    #plt.ylim((-35,5)) #change this accordingly
    # plt.plot(self.input_indices, inputs[n, :, plot_col_index],
    #          label='Inputs', marker='.', zorder=-10)
#   #Plot using original scale
    plt.plot(self.input_indices, self.inverse_function(inputs[n, :, plot_col_index], self.train_min, self.train_max),
             label='Inputs', marker='.', zorder=-10)


    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    # plt.scatter(self.label_indices, labels[n, :, label_col_index],
    #             edgecolors='k', label='Labels', c='#2ca02c', s=16)
#   #Plot using original scale
    plt.scatter(self.label_indices, self.inverse_function(labels[n, :, label_col_index], self.train_min, self.train_max),
                edgecolors='k', label='Labels', c='#2ca02c', s=16)
    
    if model is not None:
      predictions = model(inputs)
      # plt.scatter(self.label_indices, predictions[n, :, label_col_index],
      #             edgecolors='k', label='Predictions', c='#ff7f0e', s=14)
#   #Plot using original scale
      plt.scatter(self.label_indices, self.inverse_function(predictions[n, :, label_col_index], self.train_min, self.train_max),
                  edgecolors='k', label='Predictions', c='#ff7f0e', s=14)

    if n == 0:
      plt.legend()

  plt.xlabel('Time [s]')

WindowGenerator.plot = plot

# %% [markdown]
# ### 4. Create `tf.data.Dataset`s
# # Finally this `make_dataset` method will take a time series `DataFrame` and convert it to a `tf.data.Dataset` 
# of `(input_window, label_window)` pairs using the `preprocessing.timeseries_dataset_from_array` function.

# %%
def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=False,
      batch_size=self.batch_size,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset

# %% [markdown]
# The `WindowGenerator` object holds training, validation and test data. Add properties for accessing them as `tf.data.Datasets` 
# using the above `make_dataset` method. Also add a standard example batch for easy access and plotting:

# %%
@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

