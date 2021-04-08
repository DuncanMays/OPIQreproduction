# this file tests our deep q learning and OPIQ classes on the cartpole v0 env from openAI gym

import gym
import random
from matplotlib import pyplot as plt
import time
import pickle
from collections import deque
import numpy as np

from OPIQ import OPIQ_Agent
from deepQ import DeepQAgent
from neuralnets import CartpoleQnetwork
from utils import TransitionMemory

NUM_EPISODES = 300
UPDATE_INTERVAL = 5
EPS_START = 0.3
EPS_BOTTOM = 0.02
DECAY_RATE = 0.95

# some class methods are unique to opiq, and so we need a boolean switch to know weather to call them or not
testing_OPIQ = True

env = gym.make('CartPole-v0')

# agent = OPIQ_Agent(CartpoleQnetwork, 4, 2, num_steps=4)
agent = DeepQAgent(CartpoleQnetwork, 4, 2, num_steps=4)

episode_lengths = []

eps = 0.3

tm = TransitionMemory(4)

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

		# ********this is unique to OPIQ
		# this line updates the number of times the agent has visited this state/action
		if (testing_OPIQ):
			try:
				# this should be done upon the entrance of the transition to replay memory
				agent.visited(observation, action)
			except(AttributeError):
				testing_OPIQ = False

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