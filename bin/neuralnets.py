# this file contains neural nets used to approximate q values

import torch

# for the cartpolv0 env, used to test the deepQ and OPIQ classes
class CartpoleQnetwork(torch.nn.Module):

	def __init__(self, input_size, output_size):
		super(CartpoleQnetwork, self).__init__()

		self.linear1 = torch.nn.Linear(input_size, 64)
		self.linear2 = torch.nn.Linear(64, 64)
		self.linear3 = torch.nn.Linear(64, output_size)

		self.activation = torch.relu

	def forward(self, x):
		x = self.activation(self.linear1(x))
		x = self.activation(self.linear2(x))
		x = self.linear3(x)

		return x

# for the randomchain Env
class RandomChainNetwork(torch.nn.Module):

	def __init__(self, input_size, output_size):
		super(RandomChainNetwork, self).__init__()

		self.linear1 = torch.nn.Linear(input_size, 256)
		self.linear2 = torch.nn.Linear(256, 256)
		self.linear3 = torch.nn.Linear(256, output_size)

		self.activation = torch.relu

	def forward(self, x):
		x = self.activation(self.linear1(x))
		x = self.activation(self.linear2(x))
		x = self.linear3(x)

		return x

# a convolutional network for the maze env
class MazeNetwork(torch.nn.Module):

	def __init__(self, input_shape, output_size):
		super(MazeNetwork, self).__init__()

		image_size = input_shape[0]
		stride = 2
		channels = 16

		self.conv1 = torch.nn.Conv2d(1, channels, 3, stride=2)
		self.conv2 = torch.nn.Conv2d(channels, channels, 3, stride=2)

		# lil bit of code to calculate the size of the fully connected layers
		for _ in range(2):
			image_size = int((image_size + 2 * 0 - 3) / stride + 1)

		self.linear_size = image_size * image_size * channels

		self.linear1 = torch.nn.Linear(self.linear_size, self.linear_size//2)
		self.linear2 = torch.nn.Linear(self.linear_size//2, output_size)

		self.activation = torch.relu

	def forward(self, x):
		x = self.activation(self.conv1(x))
		x = self.activation(self.conv2(x))

		# flattening
		x = x.view(-1, self.linear_size)

		x = self.activation(self.linear1(x))
		x = self.linear2(x)

		return x