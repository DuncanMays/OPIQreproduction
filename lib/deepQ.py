# we want replay memory to be a transpose of transitions, if batch = replay_memory.sample(), then batch[0] should be an array of starting states
# we also want columns in replay memory to be either tensors or arrays, since we've gotta add and multiply them and it would be nice to do this in parallel
# this is a problem for terminal states, we want to use a different bellman update equation for them and so we'll need some kind of filter
# we may in need to iterate over the batch columns and set reward to itself plus the future q value if not done, this will be a serial operation, unless we maybe compile it with torch's jit?

import gym
from collections import deque
import random
import time
import pickle

import torch

REPLAY_MEMORY_SIZE = 1_000
MIN_REPLAY_MEMORY_SIZE = 100
BATCH_SIZE = 32
GAMMA = 0.99

# this implementation of deep q learning is highly specialized and integrated with this program and the environment it uses
# it cannot be used in general cases, or even exist in a separate file, because it makes many references to local variables
class DeepQAgent():

	def __init__(self, neural_architecture, observation_dim, action_dim):
		# initializing replay memory
		self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

		# initializing the neural net used for policy and bootstrapping, we will never train this model, but we will update its parameters with params from the other model
		self.model = neural_architecture(observation_dim, action_dim)

		# this is the model that will be trained on the replay memory. We will use the main model for bootstrapping, however
		self.target_model = neural_architecture(observation_dim, action_dim)
		# we want the target model and main model to have the same parameters to begin with
		self.update_model()

		# self.criterion = torch.nn.SmoothL1Loss(0.2)
		self.criterion = torch.nn.MSELoss()
		self.optimizer = torch.optim.Adam(lr=0.001, params=self.target_model.parameters())

		self.observation_dim = observation_dim
		self.action_dim = action_dim

	def get_action(self, observation):
		# converts the observation into a tensor that the neural net can operate on
		obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(dim=0)
		# selects the action corresponding to the maximal q value
		return torch.argmax(self.model(obs), dim=1)[0].item()

	def train_from_replay(self):
		# checks that the replay buffer is full enough before we start sampling from it, else the agent will focus too much on a small set of transitions
		if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
			return

		# batch should be a list of transitions
		# a transition is a tuple of: (observation, action, reward, next_observation, done)
		batch = random.sample(self.replay_memory, BATCH_SIZE)

		# transposes the batch and then separates out all the components of each transition
		# observations is a list that holds the agent's observation in each transition that was sampled from the replay buffer
		# actions is a list that holds the agent's action in each transition that was sampled from the replay buffer
		# rewards is a list that holds the reward that the agent received in each transition
		# next_observations is a list that holds the state that the state/action combination taken in the transition led to
		# state_is_terminal is a list of booleans that holds weather or not the corresponding transition was a terminal state
		(observations, actions, rewards, next_observations, state_is_terminal) = list(map(list, zip(*batch)))

		# turns everything into a tensor
		observations = torch.Tensor(observations)
		actions_tensor = torch.Tensor(actions)
		next_observations = torch.Tensor(next_observations)
		rewards = torch.Tensor(rewards)
		state_is_terminal_tensor = torch.Tensor(state_is_terminal)

		# this is a robustness alteration for when we implement n-step q learning and next_observations is a list
		if (len(next_observations.shape) > 2):
			print('fix train from replay!')
			next_observations = next_observations[:, 0]

		# gets the model's evaluations of the state, given the information available in the observation
		# note how we're using the target model, since that is the model we're training to match the observed q distribution
		q_values = self.target_model(observations)

		# this is the estimated "value" of the next state. That is, it is the model's estimated Q value for the optimal action 
		q_values_next = self.model(next_observations)

		# we now adjust the model's estimation of the q values to match the observed reward, with the bellman equation
		# clone creates a copy for us to alter to the correct value
		# detach exempts this tensor from the gradient calculation below, without this call the model's parameters would be equally 
		#  adjusted to bring it's estimates close to this figure, and to bring this figure closer to its estimates, and so it wouldn't 
		#  learn anything
		true_q_values = q_values.clone().detach()
		# range(len(true_q_values)) selects all q vectors in the transition set
		# actions selects the q value for the specific action that was taken
		true_q_values[range(len(true_q_values)), actions] = rewards + GAMMA*torch.max(q_values_next, axis=1).values

		# we now begine adjustements for terminal states
		# firstly by finding the indices of the terminal states in the batch of transitions sampled from replay memory
		terminal_indices = torch.stack(torch.where(state_is_terminal_tensor == 1.0)).tolist()[0]

		# then by setting the target q values for each prediction to only the reward, ignoring the value of the state that follows it
		true_q_values[terminal_indices, actions_tensor[terminal_indices].tolist()] = rewards[terminal_indices]

		loss = self.criterion(q_values, true_q_values)

		# clears the gradients of the network's parameters in preparation for the following update
		self.optimizer.zero_grad()
		# performs backprop (w/ autograd) to get the gradient of the loss for each parameter in the DQN's neural net
		loss.backward()
		# updates the parameters
		self.optimizer.step()

	def update_replay_memory(self, transition):
		self.replay_memory.append(transition)

	def update_model(self):
		main_params = list(self.model.parameters())
		target_params = list(self.target_model.parameters())
		for i in range(len(main_params)):
			main_params[i].data = target_params[i].clone().data

# instantiates the class for testing purposes
if(__name__ == '__main__'):
	from neuralnets import CartpoleQnetwork

	test_agent = DeepQAgent(CartpoleQnetwork, 5, 10)