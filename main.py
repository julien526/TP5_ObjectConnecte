import gym

#--------------------------------------------------------------------------------------------------------------------------------
#CARTPOLE reinforcement learning

#Observation: (Cart position, Cart Velocity, Pole angle, Pole velocity at tip)
#Action: right = 1, left = 0
#Reward return 0  if pole angle more than ±12° or Cart Position is more than ±2.4  return 1 if condition are respected
#Goal: having the highest reward count possible

env_name = "CartPole-v1"
nb_step = 100
nb_story = 10
reward_count = 0
env = gym.make(env_name)
env.seed(100)

for i_episode in range(nb_story):
    observation = env.reset()

    for t in range(300):
        env.render()
        action = 1
        observation, reward, done, info = env.step(action)
        if done:
            reward_count = t
            break
    
    print(reward_count)

env.close()