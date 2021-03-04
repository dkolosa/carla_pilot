import sys, os, glob
import numpy as np
import cv2
import time
from carla_env import Carlaenv

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

    num_episodes = 10
    iter_per_episode = 100

    # init the DL things here
    carla_env = Carlaenv()
    
    for i in range(num_episodes):
        print('begin episode')
        s = carla_env.reset()
        reward = 0
        done = False
        j = 0
        while True:
        # random action test
        # carla_env.show_cam()
            action = np.random.rand(3)

            time.sleep(1/FPS)
            s1, reward, done = carla_env.step(action)
            j += 1
            s = s1
            if done:
                print(f'Episode over {i} of {num_episodes}')
                break
