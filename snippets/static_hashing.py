import numpy as np

def get_static_hash(in_dim, out_dim):

	A = np.random.normal(size=(out_dim, in_dim))

	def hash_fn(state):
		flat = np.reshape(state, [np.size(state)])
		dot_prod = A*flat
		sign = np.sign(dot_prod)
		return str(sign.tolist())

	return hash_fn

hash_fn = get_static_hash(120, 32)

test = np.random.normal(size=[10, 2, 6])

print(hash_fn(test))