import gym
import random
import time
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------------------------------------------------------
#CARTPOLE reinforcement learning

#Observation: (Cart position, Cart Velocity, Pole angle, Pole velocity at tip)
#Action: right = 1, left = 0
#Reward return 0  if pole angle more than ±12° or Cart Position is more than ±2.4  return 1 if condition are respected
#Goal: having the highest reward count possible

env_name = "CartPole-v1"
nb_step = 100
nb_story = 10
reward_count = []
env = gym.make(env_name)
env.seed(100)

for i_episode in range(nb_story): 
    observation = env.reset() # return initial value of env

    for t in range(1000): #max number of step
        env.render() #start the simulation
        action = random.randint(0, 1) #Random action
        observation, reward, done, info = env.step(action) # return values
        
        if done: # check if simulation has failed
            reward_count.append(round(t/60, 2)) # had result into array
            break
        time.sleep(1/60)
    
    print(reward_count)

env.close()


#Print result
title = "Longest time: " + str(max(reward_count)) + "s"
plt.title(title)
plt.plot(reward_count, color="red")
plt.ylabel("Time balance")
plt.grid(True)
plt.show()

