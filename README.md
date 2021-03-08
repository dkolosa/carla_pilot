# carla_pilot
This application applies reinforcement learning on a car (agent) using the
CARLA simulator. 
The current implementation uses Twin Delayed Deep Deterministic Policy Gradients. The inputs of the neural networks is a first person (dash cam) view, and the output is the throttle, steering angle, and braking, respectively.
The goal of this project is to train to successfuly navigate the environment to get to the target destination.

The DDPG and ppo algorithms are avaliable but have not been modified to work
with carla at this time.

Feel free to fork this repo for your own projects.

