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

FPS = 30

<<<<<<< HEAD
if __name__ == '__main__':
    
    carla_env = Carlaenv()

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

    agent = TDDDPG(n_action, action_bound, layer_1_nodes, layer_2_nodes, actor_lr, critic_lr, PER, GAMMA,tau, batch_size, save_dir)

    agent.update_target_network(agent.actor, agent.actor_target, agent.tau)
    agent.update_target_network(agent.critic, agent.critic_target, agent.tau)

    load_models = False
    save = True

    # If loading model, a gradient update must be called once before loading weights
    # if load_models:
    #     load_model(agent, batch_size, env, ep, n_action, n_state)

    noise_decay = 1.0

    sum_reward = 0
    agent.sum_q = 0
    agent.actor_loss = 0
    agent.critic_loss = 0
    j = 0
    for i in range(num_episodes):
        print('begin episode')
        s = carla_env.reset()
        reward = 0
        done = False
        j = 0
        while True:

            carla_env.show_cam()

            a = np.clip(agent.actor(tf.convert_to_tensor([s], dtype=tf.float32))[0] + actor_noise()*noise_decay, a_max=action_bound,
                        a_min=-action_bound)
            time.sleep(1/FPS)

            s1, r, done = carla_env.step(a)
            # Store in replay memory
            agent.memory.add(
                (np.reshape(s, (n_state,)), np.reshape(a, (n_action,)), r, np.reshape(s1, (n_state,)), done))
            agent.train(j)

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
=======

# This sets up the environmnet, have to loop through the environment and update

if __name__ == '__main__':

    try:
        num_episodes = 201
        iter_per_episode = 100

        ENV = 'carla'
        carla_env = Carlaenv()

        model_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(os.path.join(model_dir, str(datetime.date.today()) + '-' + ENV), exist_ok=True)
        save_dir = os.path.join(model_dir, str(datetime.date.today()) + '-' + ENV)

        n_action = carla_env.action_space
        n_states = (carla_env.img_channels, carla_env.img_height, carla_env.img_width)
        action_bound = .5
        batch_size = 2
        # Will have to add conv nets for processing
        # use conv and FC layers to process the images

        # use FC layers to process the current speed
        layer_1_nodes, layer_2_nodes = 128, 128

        tau = 0.01
        actor_lr, critic_lr = 0.0001, 0.001
        GAMMA = 0.99
        ep = 0.001

        actor_noise = OrnsteinUhlenbeck(np.zeros(n_action))

        agent = TDDDPG(n_states, n_action, action_bound, layer_1_nodes, layer_2_nodes, actor_lr, critic_lr, GAMMA,tau, batch_size, save_dir)

        load_models = True
        save = True

        if load_models:
            agent.load_model()

        noise_decay = 1.0

        sum_reward = 0
        agent.sum_q = 0
        agent.actor_loss = 0
        agent.critic_loss = 0
        j = 0
        for i in range(num_episodes):
            s = carla_env.reset()

            while True:
                carla_env.show_cam()

                s_img = agent.preprocess_image(s)
                a = agent.action(s_img) + actor_noise()
                s1, r, done = carla_env.step(a)

                # Store in replay memory
                agent.memory.add(
                    (np.reshape(s, (carla_env.img_channels,carla_env.img_height, 
                    carla_env.img_width,)), np.reshape(a, (n_action,)), r, 
                    np.reshape(s1, (carla_env.img_channels,carla_env.img_height, 
                    carla_env.img_width,)), done))
                agent.train(j)

                sum_reward += r
                s = s1
                j += 1
                if done:
                    agent.save_model()
                    print(f'Episode over {i} of {num_episodes}, distance from target:{carla_env.distance:.3f} reward {r}')
                    break

        
        # clean up after your done playing
    finally:
        carla_env.cleanup()


>>>>>>> main
