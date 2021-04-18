import json
from matplotlib import pyplot as plt
import numpy as np

# env = 'cartpole'
env = 'chain'
# env = 'maze'

# fname = 'baseline.data'
fname = '1_OPIQ.data'

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

x = list(range(avg_rewards.shape[0]))

plt.plot(x, avg_rewards)
plt.show()

plt.plot(x, avg_ep_lengths)
plt.show()

