import gym
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import os
import sys
#--------------------------------------------------------------------------------------------------------------------------------
#CARTPOLE reinforcement learning

#Observation: (Cart position, Cart Velocity, Pole angle, Pole velocity at tip)
#Action: right = 1, left = 0
#Done return true  if pole angle more than ±12° or Cart Position is more than ±2.4  return false if condition are respected
#Goal: having the highest reward count possible (500 for v1 and 200 for v0)

#Credit to : JackFurby https://github.com/JackFurby/CartPole-v0

env_name = "CartPole-v1"
env = gym.make(env_name)
dirname = os.path.dirname(__file__)
print(dirname)


use_qtable = True
train_model = False
model_name = "base_model"

DISCOUNT = 0.9 # Between 0 and 1, mesure of how much we care about future reward over immediate reward
RUNS = 2000  # Number of  run
SHOW_EVERY = 500  # How often the current solution is rendered
UPDATE_EVERY = 100  # How often the current progress is recorded
START_DECAYING = 1
END_DECAYING = 4000 #RUNS // 1.3

if not train_model:
	LEARNING_RATE = 0.1
	LEARNING_RATE_DECAY = 0
	epsilon = 0.1
	epsilon_decay_value = 0
 
else:
    LEARNING_RATE = 0.5 # How much new info will override old info. 0 means nothing is learned, 1 means only most recent is considered, old knowledge is discarded
    LEARNING_RATE_DECAY = 0.0001
    epsilon =1 #not a constant, going to be decayed
    epsilon_decay_value = 0.00022


# Create bins and Q table
def create_bins_and_q_table():
	# env.observation_space.high
	# [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]
	# env.observation_space.low
	# [-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]


	numBins = 10 # number of section
	num_of_observation = len(env.observation_space.high)

	# Get the size of each bucket
	bins = [ #create range for each values
		np.linspace(-4.8, 4.8, numBins),
		np.linspace(-4, 4, numBins),
		np.linspace(-.418, .418, numBins),
		np.linspace(-4, 4, numBins)
	]

	if use_qtable:
		#reader = open("../QTable/" + model_name, "r")
		reader = open(dirname +"/QTable/" + model_name, "r")
		deserialize_data = np.array(json.loads(reader.read()))
		qTable = deserialize_data
	else:
		qTable = np.random.uniform(low=-2, high=0, size=([numBins] * num_of_observation + [env.action_space.n]))

	return bins, num_of_observation, qTable


# Given a state of the enviroment, return its descreteState index in qTable
def get_discrete_state(state, bins, num_of_observation):
	stateIndex = []
	for i in range(num_of_observation):
		stateIndex.append(np.digitize(state[i], bins[i]) -1 ) # -1 will turn bin into index
	return tuple(stateIndex)


bins, num_of_observation, qTable = create_bins_and_q_table()

previousCnt = []  # array of all scores over runs
metrics = {'ep': [], 'avg': [], 'min': [], 'max': [], 'win':[]}  # metrics recorded for graph

for run in range(RUNS):
    
	discreteState = get_discrete_state(env.reset(), bins, num_of_observation)
	done = False  # has the enviroment finished?
	cnt = 0  # how many frames as been done

	while not done:
		if run % SHOW_EVERY == 0:
			env.render()  # show the cart at chosen interval

		cnt += 1
		#Epsilon-Greedy Algorithm | exploration or exploitation
		if np.random.random() > epsilon:
			action = np.argmax(qTable[discreteState]) # Get action from Q table ** exploitation **
		else:
			action = np.random.randint(0, env.action_space.n) # Get random action ** exploration **
   
		newState, reward, done, _ = env.step(action)  # perform action on enviroment

		newDiscreteState = get_discrete_state(newState, bins, num_of_observation) # get index of futur q

		maxFutureQ = np.max(qTable[newDiscreteState])  # value of future q
		currentQ = qTable[discreteState + (action, )]  # old value

		# pole fell over / went out of bounds, negative reward
		if done and cnt < 499:
			reward = -375

		# formula to caculate all Q values
		newQ = (1 - LEARNING_RATE) * currentQ + LEARNING_RATE * (reward + DISCOUNT * maxFutureQ) #modified formula
		#newQ = currentQ + LEARNING_RATE * reward + DISCOUNT * maxFutureQ - currentQ #Basic formula
		newq_index = discreteState + (action, ) # index of new q
		qTable[newq_index] = newQ  # Update qTable with new Q value

		discreteState = newDiscreteState # update index of state

	previousCnt.append(cnt)

	# Decaying is being done every run if run number is within decaying range
	if END_DECAYING >= run >= START_DECAYING:
		epsilon -= epsilon_decay_value
		LEARNING_RATE -=LEARNING_RATE_DECAY

	# Add new metrics for graph
	if run % UPDATE_EVERY == 0:
		latestRuns = previousCnt[-UPDATE_EVERY:]
		number_of_win = latestRuns.count(500)
		averageCnt = sum(latestRuns) / len(latestRuns)
		metrics['ep'].append(run)
		metrics['avg'].append(averageCnt)
		metrics['min'].append(min(latestRuns))
		metrics['max'].append(max(latestRuns))
		metrics['win'].append(number_of_win)
		print("Run:", run, "Average:", averageCnt, "Min:", min(latestRuns), "Max:", max(latestRuns), "number_of_win", number_of_win, "Epsilon", epsilon, "Learning rate:", LEARNING_RATE)


env.close()






# Plot graph
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Results')
ax1.plot(metrics['ep'], metrics['avg'], label="average rewards")
ax1.plot(metrics['ep'], metrics['min'], label="min rewards")
ax1.plot(metrics['ep'], metrics['max'], label="max rewards")
ax1.legend(loc=4)
ax1.grid(True)
ax2.plot(range(1, RUNS//UPDATE_EVERY + 1), metrics['win'], label="Number of win")
ax2.set_xlabel('Segement of ' + str(UPDATE_EVERY) + " episodes")
ax2.grid(True)
ax2.legend(loc=4)
plt.show()



env.close()

serialize_data = json.dumps(qTable.tolist())


writer = open(dirname +"/QTable/" + model_name, "w+")
writer.write(serialize_data)





