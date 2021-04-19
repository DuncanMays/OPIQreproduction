import torch

import random
from matplotlib import pyplot as plt
import time
import numpy as np
import sys
from copy import copy

from maze_env import SnakingMaze
from maze_qnetwork import MazeNetwork

sys.path.append('../bin')
from deepQ import DeepQAgent
from OPIQ import OPIQ_Agent
from utils import permute_generator

# important constants for this trial only
MAZE_SIZE = 8
MAX_TIMESTEPS = 1_000_000
MAX_TIME_STEPS_PER_EPISODE = 250

UPDATE_INTERVAL = 1_000
time_steps_since_last_update = 0

EPS_START = 1.0
EPS_BOTTOM = 0.01

eps = EPS_START
step = (EPS_START - EPS_BOTTOM)/50_000


env = SnakingMaze(MAZE_SIZE)

def get_random_action(angent):
	return random.choice([0, 1, 2, 3])

def preprocess_observations(obs):
	observation = torch.tensor(obs)
	observation = observation.permute([2, 0, 1])
	return observation.tolist()

def run_episode(agent):
	global time_steps_since_last_update, eps

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
		if (random.uniform(0,1) < eps):
			action = get_random_action(agent)
		else:
			action = agent.get_action(observation)

		if (eps > EPS_BOTTOM):
			eps = eps - step

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

		if done or (ep_length > MAX_TIME_STEPS_PER_EPISODE):
			break

	# records the end of the episode for diagnostics
	ep_end = time.time()
	ep_time = ep_end - ep_start

	# print('total reward: '+str(total_reward)+' | episode length: '+str(ep_length)+' | time: '+str(ep_time))

	return (total_reward, ep_length, ep_time)


def training_run(agent, n_episodes):
	rewards = []
	episode_lengths = []

	num_timesteps = 0

	for i in range(n_episodes):
		print(str(num_timesteps/MAX_TIMESTEPS) + ' complete')

		(total_reward, ep_length, ep_time) = run_episode(agent)

		print(ep_time/ep_length)

		rewards.append(total_reward)
		episode_lengths.append(ep_length)

		num_timesteps += ep_length

		if (num_timesteps > MAX_TIMESTEPS):
			return (rewards, episode_lengths)

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
	M_set = [2]
	C_action_set = [0.1, 1.0, 10.0, 100.0]
	C_bootstrap_set = [0.01, 0.1, 1.0, 10.0]

	# permute_generator returns a generator object that iterates over all combinations of every list in the given tuple
	opiq_params = permute_generator((M_set, C_action_set, C_bootstrap_set))

	total_combinations = len(M_set)*len(C_action_set)*len(C_bootstrap_set)
	current_combination = 0

	for (M, C_action, C_bootstrap) in opiq_params:

		current_combination += 1
		print('testing combination '+str(current_combination)+' out of '+str(total_combinations))		

		agent = OPIQ_Agent(MazeNetwork, (10*env.size, 10*env.size, 1), 4, hash_size=128, m=M, ca=C_action, cb=C_bootstrap, batch_size=64, replay_size=250_000, min_replay_size=25_000, num_steps=3, train_on_gpu=True)

		(rewards, episode_lengths) = training_run(agent, int(MAX_TIMESTEPS/MAX_TIME_STEPS_PER_EPISODE)+1)

		head = {
			'M': M,
			'C_action': C_action,
			'C_bootstrap': C_bootstrap
		}

		save_results(str(current_combination)+'_OPIQ.data', head, rewards, episode_lengths)

def baseline_test():

	agent = DeepQAgent(MazeNetwork, (10*env.size, 10*env.size, 1), 4, batch_size=64, replay_size=250_000, min_replay_size=25_000, num_steps=3, train_on_gpu=True)

	(rewards, episode_lengths) = training_run(agent, int(MAX_TIMESTEPS/MAX_TIME_STEPS_PER_EPISODE)+1)

	head = {}

	save_results('baseline.data', head, rewards, episode_lengths)

for i in range(5):
	print('---------------------------------------------------------------------------------------')
	print(i)

	OPIQ_grid_search()

# baseline_test()
