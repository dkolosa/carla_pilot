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


# This sets up the environmnet, have to loop through the environment and update

if __name__ == '__main__':

    try:
        num_episodes = 100
        iter_per_episode = 100

        ENV = 'carla'
        # init the DL things here
        carla_env = Carlaenv()

        model_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(os.path.join(model_dir, str(datetime.date.today()) + '-' + ENV), exist_ok=True)
        save_dir = os.path.join(model_dir, str(datetime.date.today()) + '-' + ENV)

        n_action = carla_env.action_space
        n_states = carla_env.img_height * carla_env.img_width * carla_env.img_channels
        action_bound = 1
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

        load_models = False
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
                a = np.clip(agent.action(s_img) + actor_noise(), -1, 1)
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
                    print(f'Episode over {i} of {num_episodes}, reward {r}')
                    break

        
        # clean up after done
        carla_env.cleanup()

    except KeyboardInterrupt:
        carla_env.cleanup()
