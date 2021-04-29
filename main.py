import gym

env_name = "CartPole-v1"
step = 100
env = gym.make(env_name)
env.seed(100)

for i_episode in range(10):
    observation = env.reset()

    for t in range(step):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        time.sleep((1/60))
        print("State: " + str(round(t * 100 / step, 2)) + "%", flush=True, end='\r')
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print("Fininsh at " + str(t))
            break
        
        
env.close()