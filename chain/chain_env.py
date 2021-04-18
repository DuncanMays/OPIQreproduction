from copy import copy

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
			if (self.state >= self.length-1):
				reward = 1
				done = True
				self.active = False

		elif (action == 0):
			self.state = self.state - 1

			if (self.state <= 0):
				reward = 0.01
				done = True
				self.active = False

		else:
			print(str(action)+' not a valid action, must be 0 or 1')

		return (copy(self.state), reward, done, None)