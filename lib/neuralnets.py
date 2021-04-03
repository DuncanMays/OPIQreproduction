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