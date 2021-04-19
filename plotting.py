import json
from matplotlib import pyplot as plt
import numpy as np

env = 'cartpole'
# env = 'chain'
# env = 'maze'

def get_average_rewards(fname):
	global env

	f = open('./'+env+'/data/'+fname, 'r')

	training_runs = f.readlines()

	f.close()

	avg_rewards = 0
	avg_ep_lengths = 0

	print(len(training_runs))

	for run in training_runs:
		data = json.loads(run)

		rewards = np.array(data['rewards'])
		episode_lengths = np.array(data['episode_lengths'])

		if rewards.shape[0] == 250:
			avg_rewards = rewards + avg_rewards
			avg_ep_lengths = episode_lengths + avg_ep_lengths

	avg_rewards = avg_rewards/len(training_runs)
	avg_ep_lengths = avg_ep_lengths/len(training_runs)

	return avg_rewards

def get_best_hyper_params(num_param_sets):
	global env

	best_param_set = {}
	top_avg_reward = -1

	best_run_number = 0

	for i in range(num_param_sets):

		fname = str(i+1)+'_OPIQ.data'
		f = open('./'+env+'/data/'+fname, 'r')
		training_runs = f.readlines()
		f.close()

		num_runs = len(training_runs)
		print(num_runs)
		avg_reward = 0

		for run_json in training_runs:
			run = json.loads(run_json)
			avg_reward += sum(run['rewards'][len(run['rewards'])-50:])

		avg_reward = avg_reward/num_runs

		if (avg_reward > top_avg_reward):
			top_avg_reward = avg_reward
			best_param_set = json.loads(training_runs[0])['head']
			best_run_number = i

	return avg_reward, best_param_set, best_run_number

# print(get_best_hyper_params(48))

OPIQ = get_average_rewards('13_OPIQ.data')
baseline = get_average_rewards('baseline.data')

x = list(range(len(OPIQ)))

plt.plot(x, OPIQ, label='OPIQ')
plt.plot(x, baseline, label='DeepQ')

plt.ylabel('averge reward')
plt.xlabel('episode')

plt.legend()

plt.show()