from collections import deque

class TransitionMemory():

	def __init__(self, num_steps):
		self.t_deque = deque(maxlen=num_steps)

	# this function formats all transitions in memory, returns the list of them, and resets self.t_deque to an empty deque, resetting the instance to its starting state
	def flush(self):
		# formatting all transitions
		transition_list = []
		for i in range(len(self.t_deque)):
			formatted_transition = self.format(i, num_steps_till_complete=len(self.t_deque)-i-1)
			transition_list.append(formatted_transition)

		# empties the t_deque
		self.t_deque = deque(maxlen=self.t_deque.maxlen)

		return transition_list

	# this function "formats" a transition and is the core method of this class. formatting means two things:
	# 	append the next state of all the following transfers to its next state
	# 	set the last element to an integer rather than a boolean, which represents the number of transitions from this transition until the env terminates, which is useful in bootstrapping during Q learning
	# the parameters are:
	#	transition_index is the index of the transition in self.t_deque because it makes the logic of the function simpler
	# 	num_steps_till_complete is the number of transitions from this transition until the env terminates
	def format(self, transition_index, num_steps_till_complete=-1):

		# referencing self in parameters of a function is not allowed in python, so I've settled on this hack:
		if (num_steps_till_complete == -1):
			num_steps_till_complete = self.t_deque.maxlen

		# the targetted transition in the deque
		transition = self.t_deque[transition_index]

		# gotta make a new tuple with a list containing the next state of transition that states following that state will be appended to
		# the last element, which used to be a boolean that indicates weather or not the transition was terminal, will be set to an interger representing a lower bound for the number of transitions until the terminal state.
		formatted_transition = transition[:3] + ([transition[3]], num_steps_till_complete)

		for i in range(1, len(self.t_deque)):

			# this line means that i will wrap around the deque
			i = (i + transition_index) % len(self.t_deque)

			t = self.t_deque[i]
			formatted_transition[3].append(t[3])

		return formatted_transition


	# this function accepts a transition and returns a list
	# that list either contains nothing, a single transition, or all transitions in memory
	def new_transition(self, new_transition):
		# appending the new transition onto the deque
		self.t_deque.append(new_transition)

		if (len(self.t_deque) == self.t_deque.maxlen):

			if new_transition[4]:
				# if the transition is terminal, flush memory
				return self.flush()
			
			# if we've got enough transitions in memory, return the extended form of the earliest transition
			return [self.format(0)]
			
		else:
			# if we don't have enough transitions in memory, return an empty list
			return []

def make_transition(terminal = False):
	# this is a testing function, so we can import in it even though it may be called multiple times
	import numpy as np
	import random

	old_observation = np.random.uniform(size=4)
	action = random.choice([0.0, 1.0])
	reward = random.uniform(0, 1)
	observation = np.random.uniform(size=4)
	done = terminal

	return (old_observation, action, reward, observation, done)

def test_transition_memory():

	tm = TransitionMemory(5)

	print(tm.new_transition(make_transition()))
	print(tm.new_transition(make_transition()))
	print(tm.new_transition(make_transition()))
	print(tm.new_transition(make_transition()))

	print('------------------------------')

	# upon the fifth new transition, tm should contain enough tansistions to give an output
	print(tm.new_transition(make_transition())[0][3])
	print(tm.new_transition(make_transition())[0][3])
	print(tm.new_transition(make_transition())[0][3])
	print(tm.new_transition(make_transition())[0][3])
	print(tm.new_transition(make_transition())[0][3])

	print('------------------------------')

	# we now give tm a terminal transition to see how it reacts
	final_transition = make_transition(terminal=True)
	flush = tm.new_transition(final_transition)

	print(len(flush))
	print('***')
	print(flush[0][3])
	print('***')
	print(flush[1][3])
	print('***')
	print(flush[2][3])
	print('***')
	print(flush[3][3])
	print('***')
	print(flush[4][3])


def permute_generator(tuple_of_lists):

	first_list = tuple_of_lists[0]
	last_lists = tuple_of_lists[1:]
	
	if (len(last_lists) != 0):

		child_generator = permute_generator(last_lists)

		for child in child_generator:
			for element in first_list:
				yield (element, *child)

	else:
		for element in first_list:
			yield (element, )

def test_permute_generator():

	tol = ([1,2,3,4], ['a,', ' b'], ['one', 'two', 'three'])

	gen = permute_generator(tol)

	for g in gen:
		print(g)

if (__name__ == '__main__'):
	# test_transition_memory()
	test_permute_generator()
