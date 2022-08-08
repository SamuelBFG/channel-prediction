import tensorflow as tf
import pdb

class NBEATSBlock(tf.keras.layers.Layer):

	def __init__(self, input_size: int, output_size: int, block_layers: int, hidden_units: int):
		super().__init__()
		self.fc_layers = []
		for i in range(block_layers):
			self.fc_layers.append(tf.keras.layers.Dense(hidden_units, activation=tf.nn.relu))
		self.forecast = tf.keras.layers.Dense(output_size, activation=None)
		self.backcast = tf.keras.layers.Dense(input_size, activation=None)

	def call(self, x):
		inputs = x
		for layer in self.fc_layers:
			x = layer(x)
		backcast = tf.keras.activations.relu(inputs - self.backcast(x))
		return backcast, self.forecast(x)

class NBEATS(tf.keras.layers.Layer):

	def __init__(self, input_size: int, output_size: int, block_layers: int, hidden_units: int,
				num_blocks: int, block_sharing: bool):
		super().__init__()
		self.blocks = [NBEATSBlock(input_size=input_size, output_size=output_size,
						block_layers=block_layers, hidden_units=hidden_units)]
		for i in range(1, num_blocks):
			if block_sharing:
				self.blocks.append(self.blocks[0])
			else:
				self.blocks.append(NBEATSBlock(input_size=input_size, output_size=output_size,
									block_layers=block_layers, hidden_units=hidden_units))

	def call(self, x):
		level = tf.reduce_max(x, axis=-1, keepdims=True)
		backcast = tf.math.divide_no_nan(x, level)
		forecast = 0.0
		for block in self.blocks:
			backcast, forecast_block = block(backcast)
			forecast = forecast + forecast_block
		return forecast * level

# nbeats = NBEATS(input_size=12, output_size=12, block_layers=3, num_blocks=3, hidden_units=512, block_sharing=True)
# inputs = tf.random.normal([256, 12])
# forecast = nbeats(inputs)

# class NBeatsModel(tf.keras.Model):

#   def __init__(self, input_size: int, output_size: int, block_layers: int, hidden_units: int,
#   	  	  	  num_blocks: int, block_sharing: bool):
#     super(NBeatsModel, self).__init__()
#     self.nbeats = NBEATS(input_size=input_size, output_size=output_size, block_layers=block_layers, 
#     	num_blocks=num_blocks, hidden_units=hidden_units, block_sharing=block_sharing)

#   def call(self, inputs):
#     forecast = self.nbeats(inputs)
#     return forecast