import torch

def get_static_hash(in_dim, out_dim):

	A = torch.normal(mean = torch.zeros(out_dim, in_dim))

	def hash_fn(observation, action):
		observation_flat = torch.reshape(observation, [torch.numel(observation)])
		action_flat = torch.reshape(action, [torch.numel(action)])
		flat = torch.cat([observation_flat, action_flat], axis=0)
		dot_prod = A*flat
		sign = torch.sign(dot_prod)
		return hash(str(sign.tolist()))

	return hash_fn

N = get_static_hash(240, 32)

test_obs = torch.normal(mean=torch.zeros([10, 2, 6]))
test_action = torch.normal(mean=torch.zeros([6, 20]))

test_obs2 = torch.normal(mean=torch.zeros([10, 2, 6]))
test_action2 = torch.normal(mean=torch.zeros([6, 20]))

print(N(test_obs, test_action))
print(N(test_obs, test_action))
print(N(test_obs2, test_action2))