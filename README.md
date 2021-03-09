# carla_pilot
This application applies reinforcement learning on a car (agent) using the
CARLA simulator. 
The current implementation uses Twin Delayed Deep Deterministic Policy Gradients. The inputs of the neural networks is a first person (dash cam) view, and the output is the throttle, steering angle, and braking, respectively.
The goal of this project is to train to successfuly navigate the environment to get to the target destination.

The DDPG and ppo algorithms are avaliable but have not been modified to work
with carla at this time.

## Packages
This project uses python 3.7.

To install the necessary packages:
`pip install requirements.txt`

The version of carla being used is version 0.8.11

## Running
To be able to run the program, enusre that the carla package from the PythonAPI directory in CARLA_0.9.11 directory is one level outside to the carla_pilot directory, the directory structure resembles:

```
└───CARLA_0.8.11
    │   PythonAPI
        └───carla
│   
└───carla
│   
└───carla_pilot
    │   main.py
```

To begin the program, ensure that an instance is Carla is running the current working directory is carla_pilot.

Execute:

`python main.py`

Feel free to fork this repo for your own projects.

# Acknowledgements

Thank you to everyone who develops, sponsors, and contributes to the Carla project to create an amazing project.

