import argparse
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

    parser = argparse.ArgumentParser()
    parser.add_argument('-test', help='does not train the models.',
                        action="store_true")
    parser.add_argument('-load', help='load the models in the models directory',
                        action="store_true")
    parser.add_argument('-save', help='save the model in training in the models directory',
                        action="store_true")
    args = parser.parse_args()

    try:
        num_episodes = 201

        carla_env = Carlaenv()

        model_dir = os.path.join(os.getcwd(), 'models')

        n_action = carla_env.action_space
        n_states = (carla_env.img_channels, carla_env.img_height, carla_env.img_width)
        measurements = 4
        action_bound = .5
        batch_size = 2

        layer_1_nodes, layer_2_nodes = 128, 128

        tau = 0.01
        actor_lr, critic_lr = 0.0001, 0.001
        GAMMA = 0.99
        ep = 0.001

        # actor_noise = OrnsteinUhlenbeck(np.zeros(n_action))
        agent = TDDDPG(n_states, n_action, action_bound, layer_1_nodes, layer_2_nodes, actor_lr, critic_lr, GAMMA,tau, batch_size, model_dir)

        if args.load:
            agent.load_model()

        sum_reward = 0
        agent.sum_q = 0
        agent.actor_loss = 0
        agent.critic_loss = 0
        j = 0
        for i in range(num_episodes):
            s = carla_env.reset()

            while True:
                carla_env.show_cam()
                s_img = agent.preprocess_image(s[0])
                a = agent.action(s_img, s[1])
                s1, r, done = carla_env.step(a)
                # Store in replay memory
                agent.memory.add(
                    (np.reshape(s[0], (carla_env.img_channels,carla_env.img_height, 
                    carla_env.img_width,)),np.reshape(s[1], (measurements,)), np.reshape(a, (n_action,)), r, 
                    np.reshape(s1[0], (carla_env.img_channels,carla_env.img_height, 
                    carla_env.img_width,)),np.reshape(s1[1], (measurements,)),  done))
                if not args.test:
                    agent.train(j)

                sum_reward += r
                s = s1
                j += 1
                if done:
                    if args.save:
                        agent.save_model()
                    print(f'Episode {i} of {num_episodes}, distance from target:{carla_env.distance:.3f} reward {r}')
                    break

        
        # clean up after your done playing
    finally:
        carla_env.cleanup()


