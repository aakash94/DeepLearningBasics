import gym
env = gym.make('CartPole-v0')

for i_episode in range(1):
    observation = env.reset()
    for t in range(10):
        env.render()
        #print("\n\n\nTimestep :", t)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        #print("\n\nobservation\n", observation)
        #print("\n\nreward\n", reward)
        #print("\n\ndone\n", done)
        #print("\n\ninfo\n", info)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()