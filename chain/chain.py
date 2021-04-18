import torch

import random
from matplotlib import pyplot as plt
import time
import numpy as np
import sys
from copy import copy
import json

from chain_qnetwork import RandomChainNetwork
from chain_env import RandomChain

sys.path.append('../bin')
from deepQ import DeepQAgent
from OPIQ import OPIQ_Agent
from utils import permute_generator

# important constants for this trial only
CHAIN_LEN = 100
NUM_EPISODES = 150
NUM_TIMESTEPS = 109
UPDATE_INTERVAL = 200
EPSILON = 0.01

time_steps_since_last_update = 0

env = RandomChain(CHAIN_LEN)

episode_lengths = []


def get_random_action(agent):
	return random.choice([0, 1])

template = [0.0 for i in range(CHAIN_LEN)]
def preprocess_observations(index):
	observation = copy(template)
	observation[index] = 1.0
	return observation

def run_episode(agent):
	global time_steps_since_last_update

	done = False

	ep_length = 0
	total_reward = 0
	ep_start = time.time()

	# observation is a 4 vector of [position of cart, velocity of cart, angle of pole, velocity of pole at tip]
	observation = preprocess_observations(env.reset())
	old_observation = observation

	while not done:

		time_steps_since_last_update += 1
		if (time_steps_since_last_update > UPDATE_INTERVAL):
			agent.update_model()

		# env.render()
		ep_length += 1

		action = None
		if (random.uniform(0,1) < EPSILON):
			action = get_random_action(agent)
		else:
			action = agent.get_action(observation)

		# replay memory needs to know the observation that lead to the above action, so we've got to record it before we get a new observation
		old_observation = observation

		# plugs the action into state dynamics and gets a bunch of info
		observation, reward, done, debug = env.step(action)
		observation = preprocess_observations(observation)

		total_reward += reward

		# saves the transition in replay memory
		transition = (old_observation, action, reward, observation, done)

		agent.update_replay_memory(transition)

		# trains the agent on a batch from replay
		agent.train_from_replay()

		if done or (ep_length > NUM_TIMESTEPS):
			break

	# records the end of the episode for diagnostics
	ep_end = time.time()
	ep_time = ep_end - ep_start

	# print('total reward: '+str(total_reward)+' | episode length: '+str(ep_length)+' | time: '+str(ep_time))

	return (total_reward, ep_length, ep_time)


def training_run(agent, n_episodes):
	rewards = []
	episode_lengths = []

	for i in range(n_episodes):
		# print('episode number: '+str(i)+' | ', end='')

		(total_reward, ep_length, ep_time) = run_episode(agent)

		rewards.append(total_reward)
		episode_lengths.append(ep_length)

	return (rewards, episode_lengths)

def save_results(fname, head, rewards, episode_lengths, append=True):

	f = None
	if append:
		f = open('./data/'+fname, 'a+')
	else:
		f = open('./data/'+fname, 'w+')

	# writing
	to_write = {
		'head': head,
		'rewards': rewards,
		'episode_lengths': episode_lengths
	}

	f.write(json.dumps(to_write)+'\n')

	f.close()

def OPIQ_grid_search():
	M_set = [0.1, 0.5, 2.0, 10.0]
	C_action_set = [0.1, 1.0, 10.0]
	C_bootstrap_set = [0.01, 0.1, 10.0]

	# permute_generator returns a generator object that iterates over all combinations of every list in the given tuple
	opiq_params = permute_generator((M_set, C_action_set, C_bootstrap_set))

	total_combinations = len(M_set)*len(C_action_set)*len(C_bootstrap_set)
	current_combination = 0

	for (M, C_action, C_bootstrap) in opiq_params:

		current_combination += 1
		print('testing combination '+str(current_combination)+' out of '+str(total_combinations))		

		agent = OPIQ_Agent(RandomChainNetwork, 100, 2, hash_size=32, m=M, ca=C_action, cb=C_bootstrap, batch_size=64, replay_size=10_000, min_replay_size=1_000, train_on_gpu=True)

		(rewards, episode_lengths) = training_run(agent, NUM_EPISODES)

		head = {
			'M': M,
			'C_action': C_action,
			'C_bootstrap': C_bootstrap
		}

		save_results(str(current_combination)+'_OPIQ.data', head, rewards, episode_lengths)

def baseline_test():

	agent = DeepQAgent(RandomChainNetwork, 100, 2, batch_size=64, replay_size=10_000, min_replay_size=1_000, train_on_gpu=True)

	(rewards, episode_lengths) = training_run(agent, NUM_EPISODES)

	head = {}

	save_results('baseline.data', head, rewards, episode_lengths)

for i in range(5):
	print('---------------------------------------------------------------------------------------')
	print(i)

	OPIQ_grid_search()

# baseline_test()
