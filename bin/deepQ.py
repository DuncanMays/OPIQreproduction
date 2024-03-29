# we want replay memory to be a transpose of transitions, if batch = replay_memory.sample(), then batch[0] should be an array of starting states
# we also want columns in replay memory to be either tensors or arrays, since we've gotta add and multiply them and it would be nice to do this in parallel
# this is a problem for terminal states, we want to use a different bellman update equation for them and so we'll need some kind of filter
# we may in need to iterate over the batch columns and set reward to itself plus the future q value if not done, this will be a serial operation, unless we maybe compile it with torch's jit?

import gym
from collections import deque
import random
import time
import pickle

from utils import TransitionMemory

import torch

REPLAY_MEMORY_SIZE = 1_000
MIN_REPLAY_MEMORY_SIZE = 100
BATCH_SIZE = 32
GAMMA = 0.99
GPU_NAME = 'cuda:0'

# this implementation of deep q learning is highly specialized and integrated with this program and the environment it uses
# it cannot be used in general cases, or even exist in a separate file, because it makes many references to local variables
class DeepQAgent():

	def __init__(self, neural_architecture, observation_dim, num_actions,
			replay_size = REPLAY_MEMORY_SIZE,
			min_replay_size = MIN_REPLAY_MEMORY_SIZE,
			batch_size = BATCH_SIZE,
			gamma = GAMMA,
			train_on_gpu = False,
			num_steps = 1):

		# setting hyperparams
		self.REPLAY_MEMORY_SIZE = replay_size
		self.MIN_REPLAY_MEMORY_SIZE = min_replay_size
		self.BATCH_SIZE = batch_size
		self.GAMMA = gamma
		self.NUM_STEPS = num_steps
		self.TRAIN_ON_GPU = torch.cuda.is_available() and train_on_gpu

		self.observation_dim = observation_dim
		self.num_actions = num_actions

		# initializing replay memory
		self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)

		# initializing the neural net used for policy and bootstrapping, we will never train this model, but we will update its parameters with params from the other model
		self.model = neural_architecture(observation_dim, num_actions)
		# this model will stay on CPU, even if we're training on GPU since we will only be evaluating with it and then it doesn't make sense to pump things through PCI

		# this is the model that will be trained on the replay memory. We will use the main model for bootstrapping, however
		self.target_model = neural_architecture(observation_dim, num_actions)
		# moves target model to GPU if that's where we're training
		if self.TRAIN_ON_GPU: self.target_model.to(GPU_NAME)

		# we want the target model and main model to have the same parameters to begin with
		self.update_model()

		# self.criterion = torch.nn.SmoothL1Loss(0.2)
		self.criterion = torch.nn.MSELoss()
		self.optimizer = torch.optim.Adam(lr=0.001, params=self.target_model.parameters())

		# this is used to disount future q values, instead of setting it every time train_from_replay runs, I'll just set it here and reuse it
		gamma_powers =  torch.tensor([self.GAMMA**(i+1) for i in range(self.NUM_STEPS)])
		self.discount = gamma_powers.repeat(self.BATCH_SIZE).reshape([self.BATCH_SIZE, self.NUM_STEPS])

		if (self.NUM_STEPS > 1):
			# if the q-learning algorithm should take into account more than one timestep ahead, then there needs to be a transition memory
			self.tm = TransitionMemory(self.NUM_STEPS)
			self.preprocess = self.tm.new_transition
		else:
			self.preprocess = self.default_preprocess

		# this will be used to create masks to remove values from states after the env terminates
		self.mask_components = get_mask_components(self.NUM_STEPS)

	# this is the default preprocessing function for new transitions
	# it wraps the transition in a list, and changes the last element from a boolean to an integer representing the number of steps until terminal
	def default_preprocess(self, transition):
		if (transition[4]):
			# the transition is terminal
			return [transition[:3] + ([transition[3]], 0)]
		else:
			# the transition is not terminal
			return [transition[:3] + ([transition[3]], 1)]

	def get_action(self, observation):
		# converts the observation into a tensor that the neural net can operate on
		obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(dim=0)
		# gets the q-values of the actions available, given the observation
		q = self.model(obs)
		# selects the action corresponding to the maximal q value
		return torch.argmax(q, dim=len(q.shape)-1).item()

	def train_from_replay(self):
		# checks that the replay buffer is full enough before we start sampling from it, else the agent will focus too much on a small set of transitions
		if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
			return

		# batch should be a list of transitions
		# a transition is a tuple of: (observation, action, reward, next_observation, done)
		batch = random.sample(self.replay_memory, self.BATCH_SIZE)

		# transposes the batch and then separates out all the components of each transition
		# observations is a list that holds the agent's observation in each transition that was sampled from the replay buffer
		# actions is a list that holds the agent's action in each transition that was sampled from the replay buffer
		# rewards is a list that holds the reward that the agent received in each transition
		# next_observations is a list that holds the state that the state/action combination taken in the transition led to
		# steps_till_terminal is a list of integers that hold the number of steps from the transition until a terminal transition
		(observations, actions, rewards, next_observations, steps_till_terminal) = list(map(list, zip(*batch)))

		# turns everything into a tensor
		observations = torch.Tensor(observations)
		# actions is left as a list since we'll be using it as an index
		next_observations = torch.Tensor(next_observations)
		rewards = torch.Tensor(rewards)
		# steps_till_terminal is also left as a list since we'll be using it as an index

		# the first dimension of next_observations is the batch dimension, the second indexes accross NUM_STEPS timesteps
		# to feed it into self.model, we need to reshape next_observations so that it is [BATCH_SIZE*NUM_STEPS, *sample_dimensions]
		sample_dimensions = next_observations.shape[2:]
		next_observations = next_observations.view(torch.Size([self.BATCH_SIZE*self.NUM_STEPS])+sample_dimensions)

		# gets the model's evaluations of the state, given the information available in the observation
		# note how we're using the target model, since that is the model we're training to match the observed q distribution
		if self.TRAIN_ON_GPU: observations = observations.to(GPU_NAME)
		q_values = self.target_model(observations)

		# this is the estimated "value" of the next states
		q_values_next = self.model(next_observations)
		# the values of the next states, that is, the q value of the optimal action in each of them
		values_next = torch.max(q_values_next, axis=1).values
		# we now reshape values_next back to the original shape of next_observations
		values_next = values_next.view([self.BATCH_SIZE, self.NUM_STEPS])
		# this will mask off values that are past the termination of the environment
		mask = self.mask_components[steps_till_terminal]
		# this term represents the future rewards and will be used to calculate the true q values
		future_rewards = torch.sum(self.discount*mask*values_next, dim=1)

		# we now adjust the model's estimation of the q values to match the observed reward, with the bellman equation
		# clone creates a copy for us to alter to the correct value
		# detach exempts this tensor from the gradient calculation below, without this call the model's parameters would be equally 
		#  adjusted to bring it's estimates close to this figure, and to bring this figure closer to its estimates, and so it wouldn't 
		#  learn anything
		true_q_values = q_values.clone().detach()

		if self.TRAIN_ON_GPU:
			rewards = rewards.to(GPU_NAME)
			future_rewards = future_rewards.to(GPU_NAME)

		# range(len(true_q_values)) selects all q vectors in the transition set
		# actions selects the q value for the specific action that was taken
		true_q_values[range(len(true_q_values)), actions] = rewards + future_rewards

		loss = self.criterion(q_values, true_q_values)

		# clears the gradients of the network's parameters in preparation for the following update
		self.optimizer.zero_grad()
		# performs backprop (w/ autograd) to get the gradient of the loss for each parameter in the DQN's neural net
		loss.backward()
		# updates the parameters
		self.optimizer.step()

	def update_replay_memory(self, transition):
		transition_list = self.preprocess(transition)
		self.replay_memory = self.replay_memory + deque(transition_list)

	def update_model(self):
		main_params = list(self.model.parameters())
		target_params = list(self.target_model.parameters())
		for i in range(len(main_params)):
			main_params[i].data = target_params[i].cpu().clone().data

# returns a 1D tensor of size [length] with the first n elements being one and the rest being zero
def get_row(n, length):
	t = torch.zeros(size=[length])
	t[range(n)] = 1.0
	return t

# returns a 2D tensor of size [n+1, n] that has value of 1 below the diagonal and zero on and above it
def get_mask_components(n):
	t = torch.stack([get_row(i, n) for i in range(n+1)])
	return t

# instantiates the class for testing purposes
if(__name__ == '__main__'):
	# from neuralnets import CartpoleQnetwork

	# test_agent = DeepQAgent(CartpoleQnetwork, 5, 10)

	mc = get_mask_components(5)
	print(mc[[1,3,2,5,3,0]])

