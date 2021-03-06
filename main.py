import sys, os, glob
import numpy as np
import cv2
import time
import datetime
import random
from carla_env import Carlaenv
from TDDDPGtorch import TDDDPG
from utils import OrnsteinUhlenbeck

# Based on the carla tutorial script tutorial.py

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


try:
    env = Carlaenv()

    model_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(os.path.join(model_dir, str(datetime.date.today()) + '-' + 'Carla'), exist_ok=True)
    save_dir = os.path.join(model_dir, str(datetime.date.today()) + '-' + 'Carla')

    iter_per_episode = 200
    # the state is an input image and current speed (+-1)
    n_state = env.observation_space
    # the acions are the steering angle and the speed (-1, +1)
    n_action = 2
    action_bound = 40   # the max mph limit

    env.seed(1234)
    np.random.seed(1234)

    num_episodes = 1001
    PER = False

    batch_size = 1
    
    # Will have to add conv nets for processing
    # use conv and FC layers to process the images

    # use FC layers to process the current speed
    layer_1_nodes, layer_2_nodes = 512, 512

    tau = 0.01
    actor_lr, critic_lr = 0.0001, 0.001
    GAMMA = 0.99
    ep = 0.001

    actor_noise = OrnsteinUhlenbeck(np.zeros(n_action))

    agent = TDDDPG(n_action, action_bound, layer_1_nodes, layer_2_nodes, actor_lr, critic_lr, GAMMA,tau, batch_size, save_dir)

    agent.update_target_network(agent.actor, agent.actor_target, agent.tau)
    agent.update_target_network(agent.critic, agent.critic_target, agent.tau)

    load_models = False
    save = True

    # If loading model, a gradient update must be called once before loading weights
    # if load_models:
    #     load_model(PER, agent, batch_size, env, ep, n_action, n_state)

    noise_decay = 1.0

    sum_reward = 0
    agent.sum_q = 0
    agent.actor_loss = 0
    agent.critic_loss = 0
    j = 0
    for i in range(num_episodes):
        s = env.reset()
        while True:
            # env.render()

            a = np.clip(agent.actor(tf.convert_to_tensor([s], dtype=tf.float32))[0] + actor_noise()*noise_decay, a_max=action_bound,
                        a_min=-action_bound)
            s1, r, done, _ = env.step(a)
            # Store in replay memory
            agent.memory.add(
                (np.reshape(s, (n_state,)), np.reshape(a, (n_action,)), r, np.reshape(s1, (n_state,)), done))
            # agent.train(j)

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

finally:
    print('destroy')
#     env.destroy()

# def load_model(PER, agent, batch_size, env, ep, n_action, n_state):
#     for i in range(batch_size + 1):
#         s = env.reset()
#         a = agent.actor(tf.convert_to_tensor([s], dtype=tf.float32))[0]
#         s1, r, done, _ = env.step(a)
#         # Store in replay memory
#         if PER:
#             error = abs(r + ep)  # D_i = max D
#             agent.memory.add(error, (
#                 np.reshape(s, (n_state,)), np.reshape(a, (n_action,)), r, np.reshape(s1, (n_state,)), done))
#         else:
#             agent.memory.add(
#                 (np.reshape(s, (n_state,)), np.reshape(a, (n_action,)), r, np.reshape(s1, (n_state,)), done))
#     agent.train()
#     agent.load_model()