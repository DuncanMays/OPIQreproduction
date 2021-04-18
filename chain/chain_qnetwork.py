import torch

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