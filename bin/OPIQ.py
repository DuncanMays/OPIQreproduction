from collections import deque, defaultdict
import random
import numpy as np
from deepQ import DeepQAgent

import torch

# deep Q hyper-params
REPLAY_MEMORY_SIZE = 1_000
MIN_REPLAY_MEMORY_SIZE = 100
BATCH_SIZE = 32
GAMMA = 0.99
GPU_NAME = 'cuda:0'

# OPIQ hyper-params
M = 0.5
C_action = 1
C_bootstrap = 0.1
HASH_SIZE = 32

# whre the magic of OPIQ happens
class OPIQNoveltyModule(torch.nn.Module):

	def __init__(self, input_length, hash_size, m):
		super(OPIQNoveltyModule, self).__init__()

		self.M = m

		# keeps track of state/action pseudocounts
		self.count_dict = defaultdict(lambda : 0)

		# used to calculate hashes for state/action pseudocounts
		self.A = torch.normal(0, 1, size=torch.Size([input_length, hash_size]))
		self.b = torch.normal(0, 1, size=torch.Size([hash_size]))
		self.flatten = torch.nn.Flatten()

	def forward(self, states, actions):

		# flattens state and concatenates it with actions into one, long tensor
		actions_flat = actions.unsqueeze(dim=1)
		states_flat = self.flatten(states)
		flat = torch.cat([states_flat, actions_flat], axis=1).T

		# multiplies the long tensor containing elements from states and actions by the random matrix, and gets the sign of all elements
		binary = torch.sign(torch.matmul(self.A, flat) + self.b).T

		# I hate this line of code
		# It is by far the slowest part of the program, since it represents many serial computations, not to mention it being overly complicated
		counts = torch.tensor([self.count_dict[hash(tuple(binary.tolist()[i]))] for i in range(binary.shape[0])])

		# applies the OPIQ equation to return the novelty score
		novelty_score = 1/((counts + 1)**self.M)

		return novelty_score

	# records a visit to a state/action
	def add_visit(self, state, action):
		# makes a tensor out of the state/action
		flat = torch.cat([torch.tensor(state, dtype=torch.float32).flatten(), torch.tensor(action).unsqueeze(dim=0)], axis=0)
		# turns it into a binary array hash
		binary = torch.sign(torch.matmul(self.A, flat))
		# hashes that binary array so we can use it as a key to count_dict
		state_action_hash = hash(tuple(binary.tolist()))
		# increments the cound of this state/action
		self.count_dict[state_action_hash] += 1


	# as the name suggests this resets the count dict to zero in all state/actions
	def reset_count(self):
		self.count_dict = defaultdict(lambda : 0)

class OPIQ_Agent(DeepQAgent):

	def __init__(self, neural_architecture, observation_dim, num_actions,
			replay_size = REPLAY_MEMORY_SIZE,
			min_replay_size = MIN_REPLAY_MEMORY_SIZE,
			batch_size = BATCH_SIZE,
			gamma = GAMMA,
			m = M,
			ca = C_action,
			cb = C_bootstrap,
			hash_size = HASH_SIZE,
			train_on_gpu = False,
			num_steps = 1):

		self.BATCH_SIZE = batch_size
		self.M = m
		self.C_action = ca
		self.C_bootstrap = cb
		self.HASH_SIZE = hash_size
		self.NUM_STEPS = num_steps

		input_length = 1
		try:
			for dim in observation_dim:
				input_length = input_length * dim
		except(TypeError):
			input_length =  observation_dim

		self.novely_module = OPIQNoveltyModule(self.HASH_SIZE, input_length+1, self.M)

		# this is used to disount future OPIQ novelty scores
		OPIQ_bootstrap_powers =  torch.tensor([self.C_bootstrap**(i+1) for i in range(self.NUM_STEPS)])
		self.OPIQ_discount = OPIQ_bootstrap_powers.repeat(self.BATCH_SIZE).reshape([self.BATCH_SIZE, self.NUM_STEPS])

		super(OPIQ_Agent, self).__init__(neural_architecture, observation_dim, num_actions, replay_size = replay_size, min_replay_size = min_replay_size, batch_size = batch_size, gamma = gamma, train_on_gpu = train_on_gpu, num_steps = num_steps)

	def get_action(self, observation):
		# converts the observation into a tensor that the neural net can operate on
		obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(dim=0)

		# gets the predicted q values as given by the neural net, and 
		q_vals = self.model(obs)
		
		# the novelty of eacha action, as predicted by OPIQ
		novelty = torch.stack([self.novely_module(obs, torch.tensor([action])) for action in range(self.num_actions)])

		print(novelty.shape)
		print(q_vals.shape)

		# selects the action as the maximum of a weighted combination of the action's q value and novelty
		score = q_vals + self.C_action*novelty
		action = torch.argmax(score, dim=len(score.shape)-1)[0].item()

		return action

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

		# these re the q value of the next states, as estimated by our model
		q_values_next = self.model(next_observations)
		# the actions with maximal q value, that would be taken in those states should the agent reach them with the current model
		next_actions = torch.max(q_values_next, axis=len(q_values_next.shape)-1).indices.flatten()
		# the values of the next states, that is, the q value of the optimal action in each of them
		values_next = q_values_next[range(self.BATCH_SIZE*self.NUM_STEPS), next_actions]
		# we now reshape values_next back to the original shape of next_observations
		values_next = values_next.view([self.BATCH_SIZE, self.NUM_STEPS])
		# this will mask off values that are past the termination of the environment
		mask = self.mask_components[steps_till_terminal]
		# this term represents the future value and will be used to calculate the true q values
		future_value = torch.sum(self.discount*mask*values_next, dim=1)

		# we now adjust the model's estimation of the q values to match the observed reward, with the bellman equation
		# clone creates a copy for us to alter to the correct value
		# detach exempts this tensor from the gradient calculation below, without this call the model's parameters would be equally 
		#  adjusted to bring it's estimates close to this figure, and to bring this figure closer to its estimates, and so it wouldn't 
		#  learn anything
		true_q_values = q_values.clone().detach()

		# we should calculate OPIQ stuff here, that being calculating future novelties
		# like with values_next, we've gotta view it back into shape
		future_novelty = self.novely_module(next_observations, next_actions).view([self.BATCH_SIZE, self.NUM_STEPS])
		# this term represents the novelty of future state/actions (according to OPIQ) and will be used to calculate the true q values. It gets the same mask as the genuine rewards
		future_novelty_value = torch.sum(self.OPIQ_discount*mask*values_next, dim=1)

		if self.TRAIN_ON_GPU:
			rewards = rewards.to(GPU_NAME)
			future_value = future_value.to(GPU_NAME)
			future_novelty_value = future_novelty_value.to(GPU_NAME)

		# range(len(true_q_values)) selects all q vectors in the transition set
		# actions selects the q value for the specific action that was taken
		true_q_values[range(len(true_q_values)), actions] = rewards + future_value + future_novelty_value

		loss = self.criterion(q_values, true_q_values)

		# clears the gradients of the network's parameters in preparation for the following update
		self.optimizer.zero_grad()
		# performs backprop (w/ autograd) to get the gradient of the loss for each parameter in the DQN's neural net
		loss.backward()
		# updates the parameters
		self.optimizer.step()

	def update_replay_memory(self, transition):
		# records the visit to the state before putting it in replay, note that this happens before the transition is preprocessed
		observation = transition[0]
		action = transition[1]
		self.novely_module.add_visit(observation, action)

		super(OPIQ_Agent, self).update_replay_memory(transition)


# instantiates the class for testing purposes
if(__name__ == '__main__'):
	from neuralnets import CartpoleQnetwork

	test_agent = OPIQ_Agent(CartpoleQnetwork, 5, 10)