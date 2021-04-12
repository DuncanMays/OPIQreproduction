import torch

import random
from matplotlib import pyplot as plt
import time
import numpy as np
import sys
from copy import copy

sys.path.append('../lib')
from deepQ import DeepQAgent
from neuralnets import CartpoleQnetwork

from env import RandomChain

# important constants for this trial only
CHAIN_LEN = 100
NUM_EPISODES = 250
UPDATE_INTERVAL = 5
EPS_START = 0.3
EPS_BOTTOM = 0.02
DECAY_RATE = 0.95

env = RandomChain(CHAIN_LEN)

# baseline
agent = DeepQAgent(CartpoleQnetwork, CHAIN_LEN, 2, num_steps=1)
# the weird one
# agent = DeepQAgent(CartpoleQnetwork, 100, 2, batch_size=64, gamma=0.5, num_steps=3, train_on_gpu=True)

episode_lengths = []

eps = 0.3

def get_random_action():
	return random.choice([0, 1])

# obs will 
template = [0.0 for i in range(CHAIN_LEN)]
def preprocess_observations(obs):
	observation = copy(template)
	observation[obs] = 1.0
	return observation


for i in range(NUM_EPISODES):
	# records the start of the episode for diagnostics
	ep_start = time.time()

	# this will be set to true when an episode ends, so we've gotta reset it here
	done = False

	# the length of the episode, reset to zero
	ep_length = 0

	# observation is a 4 vector of [position of cart, velocity of cart, angle of pole, velocity of pole at tip]
	observation = preprocess_observations(env.reset())
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
			action = get_random_action()
		else:
			action = agent.get_action(observation)

		# replay memory needs to know the observation that lead to the above action, so we've got to record it before we get a new observation
		old_observation = observation

		# plugs the action into state dynamics and gets a bunch of info
		observation, reward, done, debug = env.step(action)
		observation = preprocess_observations(observation)

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


x = [i for i in range(len(episode_lengths))]
plt.plot(x, episode_lengths)
plt.show()

