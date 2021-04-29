import torch as T
import numpy as np
import gym
import gym.spaces
import os, datetime
from utils import OrnsteinUhlenbeck
from DDPGtorch import DDPG

def test_rl():
    """Test the RL algorithm using an openai gym environment"""

    ENVS = ('Pendulum-v0', 'MountainCarContinuous-v0', 'BipedalWalker-v3', 'LunarLanderContinuous-v2',
        'BipedalWalkerHardcore-v3')

    ENV = ENVS[0]
    
    model_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(os.path.join(model_dir, str(datetime.date.today()) + '-' + ENV), exist_ok=True)
    save_dir = os.path.join(model_dir, str(datetime.date.today()) + '-' + ENV)

    env = gym.make(ENV)
    iter_per_episode = 200

    n_state = env.observation_space.shape
    n_action = env.action_space.shape[0]
    action_bound = 1

    env.seed(1234)
    np.random.seed(1234)

    num_episodes = 1001
    PER = False

    batch_size = 128
    #Pendulum
    layer_1_nodes, layer_2_nodes = 128, 128

    tau = 0.001
    actor_lr, critic_lr = 0.001, 0.0001
    GAMMA = 0.99
    ep = 0.001

    actor_noise = OrnsteinUhlenbeck(np.zeros(n_action))

    agent = DDPG(n_state, n_action, action_bound, layer_1_nodes, layer_2_nodes, actor_lr, critic_lr, PER, GAMMA,
                 tau, batch_size, save_dir)

    agent.update_target_network(tau)

    load_models = False
    save = True

    # If loading model, a gradient update must be called once before loading weights
    if load_models:
        load_model(agent)

    for i in range(num_episodes):
        s = env.reset()
        sum_reward = 0
        agent.sum_q = 0
        agent.actor_loss = 0
        agent.critic_loss = 0

        while True:
            env.render()
            a = agent.action(s)
            a_clip = a + actor_noise()

            s1, r, done, _ = env.step(a_clip)

            # Store in replay memory
            if PER:
                error = 1 # D_i = max D
                agent.memory.add(error, (
                    (np.reshape(s, (n_state[0],)), np.reshape(a_clip, (n_action,)), r, np.reshape(s1, (n_state[0],)), done)))
            else:
                agent.memory.add(
                    (np.reshape(s, (n_state[0],)), np.reshape(a_clip, (n_action,)), r, np.reshape(s1, (n_state[0],)), done))
            agent.train()

            sum_reward += r
            s = s1
            if done:
                print(f'Episode: {i}, reward: {int(sum_reward)}')
                # rewards.append(sum_reward)
                print('===========')
                if save:
                    agent.save_model()
                break

def load_model(agent):
    agent.load_model()


if __name__ == '__main__':
    # with tf.device('/CPU:0'):
    test_rl()
