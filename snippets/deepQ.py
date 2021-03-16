# we want replay memory to be a transpose of transitions, if batch = replay_memory.sample(), then batch[0] should be an array of starting states
# we also want columns in replay memory to be either tensors or arrays, since we've gotta add and multiply them and it would be nice to do this in parallel
# this is a problem for terminal states, we want to use a different bellman update equation for them and so we'll need some kind of filter
# we may in need to iterate over the batch columns and set reward to itself plus the future q value if not done, this will be a serial operation, unless we maybe compile it with torch's jit?

import gym
from collections import deque
import random
from matplotlib import pyplot as plt
import time
import pickle

import torch

env = gym.make('CartPole-v0')

REPLAY_MEMORY_SIZE = 1_000
MIN_REPLAY_MEMORY_SIZE = 100
BATCH_SIZE = 32
NUM_EPISODES = 300
GAMMA = 0.99
EPS_START = 0.3
EPS_BOTTOM = 0.02
DECAY_RATE = 0.95
UPDATE_INTERVAL = 5

# the neural net used to approximate q values
class CartpoleQnetwork(torch.nn.Module):

	def __init__(self):
		super(CartpoleQnetwork, self).__init__()

		self.linear1 = torch.nn.Linear(4, 64)
		self.linear2 = torch.nn.Linear(64, 64)
		self.linear3 = torch.nn.Linear(64, 2)

		self.activation = torch.relu

	def forward(self, x):
		x = self.activation(self.linear1(x))
		x = self.activation(self.linear2(x))
		x = self.linear3(x)

		return x

# this implementation of deep q learning is highly specialized and integrated with this program and the environment it uses
# it cannot be used in general cases, or even exist in a separate file, because it makes many references to local variables
class DeepQAgent():

	def __init__(self):
		# initializing replay memory
		self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

		# initializing the neural net used for policy and bootstrapping, we will never train this model, but we will update its parameters with params from the other model
		self.model = CartpoleQnetwork()

		# this is the model that will be trained on the replay memory. We will use the main model for bootstrapping, however
		self.target_model = CartpoleQnetwork()
		# we want the target model and main model to have the same parameters to begin with
		self.update_model()

		# self.criterion = torch.nn.SmoothL1Loss(0.2)
		self.criterion = torch.nn.MSELoss()
		self.optimizer = torch.optim.Adam(lr=0.001, params=self.target_model.parameters())

	def get_action(self, observation):
		# converts the observation into a tensor that the neural net can operate on
		obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(dim=0)
		# gets the argmax of the model's output on obs and then converts it into an integer corersponding to a certain action
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
		# range(len(true_q_values)) selects all q vectors in the batch
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

agent = DeepQAgent()

episode_lengths = []

eps = 0.3

for i in range(NUM_EPISODES):
	# records the start of the episode for diagnostics
	ep_start = time.time()

	# this will be set to true when an episode ends, so we've gotta reset it here
	done = False

	# the length of the episode, reset to zero
	ep_length = 0

	# observation is a 4 vector of [position of cart, velocity of cart, angle of pole, velocity of pole at tip]
	observation = env.reset()
	old_observation = observation

	if (eps > EPS_BOTTOM):
		eps = eps*DECAY_RATE

	if (i%UPDATE_INTERVAL == 0):
		agent.update_model()

	while not done:

		# env.render()
		ep_length += 1

		action = None
		if (random.uniform(0,1) < eps):
			action = env.action_space.sample()
		else:
			action = agent.get_action(observation)

		# replay memory needs to know the observation that lead to the above action, so we've got to record it before we get a new observation
		old_observation = observation

		# plugs the action into state dynamics and gets a bunch of info
		observation, reward, done, debug = env.step(action)

		# saves the transition in replay memory
		transition = (old_observation, action, reward, observation, done)
		agent.update_replay_memory(transition)

		# trains the agent on a batch from replay
		agent.train_from_replay()

		if done:
			break

	# records the end of the episode for diagnostics
	ep_end = time.time()
	ep_time = ep_end - ep_start

	print('episode: '+str(i)+' | length: '+str(ep_length)+' | epsilon: '+str(round(100*eps, 1))+' | time(ms): '+str(round(1000*ep_time, 1)))
	episode_lengths.append(ep_length)


# saves model parameters
# print('saving model params to disk')
# params_list = list(agent.model.parameters())
# byte_strm = pickle.dumps(params_list)
# f = open(time.asctime(), 'wb')
# f.write(byte_strm)
# f.close()

x = [i for i in range(len(episode_lengths))]
plt.plot(x, episode_lengths)
plt.show()