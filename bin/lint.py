# this implementation of deep q learning is highly specialized and integrated with this program and the environment it uses
# it cannot be used in general cases, or even exist in a separate file, because it makes many references to local variables
class OPIQ_Agent(DeepQAgent):

	def __init__(self, neural_architecture, observation_dim, action_dim):
		super(OPIQ_Agent, self).__init__(neural_architecture, observation_dim, action_dim)

		# this matrix will be used to hash state/action pairs
		self.A = torch.normal(mean = torch.zeros(32, 5))
		# this dict will hold the number of times each state/action has been visited, with the hashes as keys
		self.novelty_dict = {}
		# this param controls the rate of descent of the novelty score wrt the number of visits to each state, higher means novelty decays faster, lower means novelty decays slower
		self.novelty_decay = M

	def get_action(self, observation):
		# converts the observation into a tensor that the neural net can operate on
		obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(dim=0)
		# gets the predicted q values as given by the neural net, and 
		q_vals = self.model(obs)
		# gets the novelty score of eacha action
		novelty = self.novelty_tensor(observation)
		# selects the action as the maximum of a weighted combination of the action's q value and novelty
		action = torch.argmax(q_vals + C_action*novelty, dim=1)[0].item()

		return action

	def get_num_visits(self, state, action):
		n = None
		try:
			n = self.novelty_dict[self.hash_fn(state, action)]
		except(KeyError):
			n = 0
			self.novelty_dict[self.hash_fn(state, action)] = 0
		return n

	def hash_fn(self, observation, action):
		observation = torch.tensor(observation)
		action = torch.tensor(action)
		observation_flat = torch.reshape(observation, [torch.numel(observation)])
		action_flat = torch.reshape(action, [torch.numel(action)])
		flat = torch.cat([observation_flat, action_flat], axis=0)
		dot_prod = self.A*flat
		sign = torch.sign(dot_prod)
		return hash(str(sign.tolist()))

	def novelty_score(self, state, action):
		n = self.get_num_visits(state, action)
		return 1/((n+1)**self.novelty_decay)

	# calculates the novelty scores of all actions in a state, and returns it as a tensor.
	# does not multiply by C factor, as that is left for the caller
	def novelty_tensor(self, state):
		scores = []
		for action in range(self.action_dim):
			scores.append(self.novelty_score(state, action))
		return torch.tensor(scores)
		

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
		# range(len(true_q_values)) selects all q vectors in the transition set
		# actions selects the q value for the specific action that was taken
		# ***we must get the actions that were taken in the future steps and use them in OPIQ calculation
		true_q_values[range(len(true_q_values)), actions] = rewards + GAMMA*torch.max(q_values_next, axis=1).values

		# we now begine adjustements for terminal states
		# firstly by finding the indices of the terminal states in the batch of transitions sampled from replay memory
		terminal_indices = torch.stack(torch.where(state_is_terminal_tensor == 1.0)).tolist()[0]

		# then by setting the target q values for each prediction to only the reward, ignoring the value of the state that follows it
		true_q_values[terminal_indices, actions_tensor[terminal_indices].tolist()] = rewards[terminal_indices]

		# we now adjust the q values with the novelty score, this is a crucial part of OPIQ and is unique to it
		future_novelty = torch.stack([self.novelty_tensor(observation) for observation in observations])
		true_q_values = true_q_values + C_bootstrap*future_novelty

		loss = self.criterion(q_values, true_q_values)

		# clears the gradients of the network's parameters in preparation for the following update
		self.optimizer.zero_grad()
		# performs backprop (w/ autograd) to get the gradient of the loss for each parameter in the DQN's neural net
		loss.backward()
		# updates the parameters
		self.optimizer.step()

	# counts the number of visits to each state
	def visited(self, state, action):
		try:
			self.novelty_dict[self.hash_fn(state, action)] += 1
		except(KeyError):
			self.novelty_dict[self.hash_fn(state, action)] = 1
