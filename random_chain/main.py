from copy import copy

import torch

# common accross all trials
GAMMA = 0.99
OPTIMIZER = torch.optim.RMSprop
LEARNING_RATE = 0.005

# important constants for this trial only
CHAIN_LEN = 100


class RandomChain():

	def __init__(self, length):
		self.length = length
		self.active = False
		self.state = None

	def reset(self):
		self.active = True
		self.state = 1
		return copy(self.state)

	def step(self, action):
		# checks that the environment doesn't need to be reset
		if (self.active == False):
			print('environment not active')
			return

		done = False
		reward = 0

		if (action == 1):
			self.state = self.state + 1

			# if the agent has reached the end of the chain
			if (self.state == self.length):
				reward = 1
				done = True
				self.active = False

		elif (action == 0):
			self.state = self.state - 1

			if (self.state == 0):
				reward = 0.01
				done = True
				self.active = False

		else:
			print('action not valid: '+str(action))

		return (copy(self.state), reward, done)

# turns the state of a chain env into a one-hot tensor that can be fed into a neural net
def one_hot_chain_env(state):
	t = torch.zeros(CHAIN_LEN+1)
	t[state] = 1.0
	return t

env = RandomChain(5)



