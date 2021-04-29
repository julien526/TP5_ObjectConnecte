import gym

env_name = "CartPole-v1"
nb_step = 100
nb_story = 10
env = gym.make(env_name)
env.seed(100)

for i_episode in range(10):
    observation = env.reset()

    for t in range(nb_step):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.nb_step(action)
        time.sleep((1/60))
        print("State: " + str(round(t * 100 / nb_step, 2)) + "%", flush=True, end='\r')
        if done:
            print("Episode finished after {} timenb_steps".format(t+1))
            print("Fininsh at " + str(t))
            break
        
env.close()