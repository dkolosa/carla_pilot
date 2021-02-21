import tensorflow as tf
import numpy as np
import gym
import gym.spaces
import os, datetime
from utils import OrnsteinUhlenbeck
from DDPG import DDPG


def test_rl():
    """Test the RL algorithm using an openai gym environment"""

    ENV = 'CarRacing-v0' # uses pixels

    model_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(os.path.join(model_dir, str(datetime.date.today()) + '-' + ENV), exist_ok=True)
    save_dir = os.path.join(model_dir, str(datetime.date.today()) + '-' + ENV)

    env = gym.make(ENV)
    iter_per_episode = 200

    # input image (96,96,3)
    img_height = env.observation_space.shape[0]
    img_width = env.observation_space.shape[1]
    n_channels = env.observation_space.shape[2]

    n_state = env.observation_space.shape
    n_action = 3

    action_bound = 1

    env.seed(1234)
    np.random.seed(1234)

    num_episodes = 1001
    PER = False

    batch_size = 12
    #Pendulum
    layer_1_nodes, layer_2_nodes = 256, 256

    tau = 0.01
    actor_lr, critic_lr = 0.0001, 0.001
    GAMMA = 0.99
    ep = 0.001

    actor_noise = OrnsteinUhlenbeck(np.zeros(n_action))

    agent = DDPG(n_action, action_bound, layer_1_nodes, layer_2_nodes, actor_lr, critic_lr, PER, GAMMA,
                 tau, batch_size, save_dir)

    agent.update_target_network(agent.actor, agent.actor_target, agent.tau)
    agent.update_target_network(agent.critic, agent.critic_target, agent.tau)

    load_models = False
    save = False

    # If loading model, a gradient update must be called once before loading weights
    if load_models:
        load_model(PER, agent, batch_size, env, ep, n_action, n_state)

    for i in range(num_episodes):
        s = env.reset()
        sum_reward = 0
        agent.sum_q = 0
        agent.actor_loss = 0
        agent.critic_loss = 0
        j = 0

        while True:
            env.render()
            s = agent.preprocess(s)
            
            a = tf.squeeze(agent.actor(s))
            a_clip = np.clip(tf.squeeze(a),-1, 1)

            s1, r, done, _ = env.step(a_clip)

            # Store in replay memory
            if PER:
                error = 1 # D_i = max D
                agent.memory.add(error, (
                np.reshape(s, (n_state,)), np.reshape(a_clip, (n_action,)), r, np.reshape(s1, (n_state,)), done))
            else:
                agent.memory.add(
                    (np.reshape(s, (img_height*img_width*n_channels,)), np.reshape(a_clip, (n_action,)), r, np.reshape(s1, (img_width*img_height*3,)), done))
            agent.train()

            sum_reward += r
            s = s1
            j += 1
            if done:
                print(f'Episode: {i}, reward: {int(sum_reward)}, q_max: {agent.sum_q / float(j)},\nactor loss:{agent.actor_loss / float(j)}, critic loss:{agent.critic_loss/ float(j)}')
                # rewards.append(sum_reward)
                print('===========')
                if save:
                    agent.save_model()
                if sum_reward > 0:
                    noise_decay = 0.001
                break

def load_model(PER, agent, batch_size, env, ep, n_action, n_state):
    for i in range(batch_size + 1):
        s = env.reset()
        a = agent.actor(tf.convert_to_tensor([s], dtype=tf.float32))[0]
        s1, r, done, _ = env.step(a)
        # Store in replay memory
        if PER:
            error = abs(r + ep)  # D_i = max D
            agent.memory.add(error, (
                np.reshape(s, (n_state,)), np.reshape(a, (n_action,)), r, np.reshape(s1, (n_state,)), done))
        else:
            agent.memory.add(
                (np.reshape(s, (n_state,)), np.reshape(a, (n_action,)), r, np.reshape(s1, (n_state,)), done))
    agent.train()
    agent.load_model()


if __name__ == '__main__':
    # with tf.device('/CPU:0'):
    test_rl()
