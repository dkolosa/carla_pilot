import os, sys, glob, time
import numpy as np
import random
import time
import cv2
import math
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

class Carlaenv():
    def __init__(self):

        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(2.0)

        # Image height and width
        self.img_width = 480
        self.img_height = 480
        self.img_channels = 3
        self.inpuut_image = None
        self.int_step = 0
        self.actor_list = []

        self.dash_cam = None

        self.observation_space = (self.img_height, self.img_width)
        self.action_space = 3

        # Once we have a client we can retrieve the world that is currently
        self.world = self.client.get_world()

        # The world contains the list blueprints that we can use for adding new
        blueprint_library = self.world.get_blueprint_library()

        self.vehicle_bp = random.choice(blueprint_library.filter('vehicle.toyota.*'))

        # The destination will be another spawn point on the map
        self.destination = random.choice(self.world.get_map().get_spawn_points())
        self.start_point = random.choice(self.world.get_map().get_spawn_points())
        self.destination_loc = self.destination.location
        start_dist = self.start_point.location
        self.dest = [self.destination_loc.x, self.destination_loc.y]
        self.dist_norm = np.sqrt((start_dist.x-self.dest[0])**2 + (start_dist.y-self.dest[1])**2)
        self.start_time = None
        self.prev_accel = 0.0
        self.prev_steer = 0.0
        self.distance = 0

    def setup_sensors(self):
        # setup a dashcam sensor
        # Find the blueprint of the sensor.
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute('image_size_x', f'{self.img_width}')
        self.camera_bp.set_attribute('image_size_y', f'{self.img_height}')
        self.camera_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        # self.camera_bp.set_attribute('sensor_tick', '1.0') 
        # move the camera to the dash of the car
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.sensor_cam = self.world.spawn_actor(self.camera_bp, camera_transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor_cam)

    def step(self, action):
        # process action
        speed = round(float(action[0]),3)
        steering_angle = round(float(action[1]),3)
        braking = 0
        # Will have to scale/normalize the actions
        self.vehicle.apply_control(carla.VehicleControl(throttle=speed, steer=steering_angle, brake=braking))

        # state is the image
        position = self.vehicle.get_transform() # given in meters
        vel_vec = self.vehicle.get_velocity()  # m/s
        vel = math.sqrt(vel_vec.x**2 + vel_vec.y**2 + vel_vec.z**2)
        accel_vec = self.vehicle.get_acceleration() # m/s^2
        accel = math.sqrt(accel_vec.x**2 + accel_vec.y**2 + accel_vec.z**2)
        delta_accel = accel - self.prev_accel
        self.distance = np.sqrt((self.dest[0]-position.location.x)**2 +
                     (self.dest[1]-position.location.y)**2)
        # calculate reward
        reward, done = self.reward(self.distance, action, delta_accel)
        measurements = [position.location.x, position.location.y, vel_vec.x, vel_vec.y, accel_vec.x, accel_vec.y]          
        self.prev_steer = steering_angle
        return self.dash_cam, reward, done

    def reward(self, distance, action, accel_cheange):
        '''The reward signal takes the current state of the agent into account.
        The agent is penalized for bad driving habits (hard braking, high acceleration,
        sharp turns, driving too close to other vehicles)'''

        speed, steering, brake = action
        reward_col = 0
        done = False

        delta_steering = abs(steering-self.prev_steer)

        # check for collision
        # self.sensor_collision.listen(lambda data: check_collision(data))
        if self.collision:
            self.collision = False
            reward_col = 1
            done = True

        # reward = destination + speed_limit_threshold - break_force - collisions - change streeing angle -
                    # jerk + distance_from_cars_threshold
        reward = -(distance/self.dist_norm) - reward_col - accel_cheange*10 - delta_steering*10

        if -.1 <= distance <= .1:
            reward = 1
            done = True

        return reward, done

    def reset(self):
        self.cleanup()
        self.actor_list = []
        start = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.vehicle_bp, start)
        self.collision = False
        self.setup_sensors()
        self.sensor_cam.listen(lambda data: self.process_image(data))
        self.actor_list.append(self.vehicle)
        self.int_step = 0

        # setup the collision sensor
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        collision_transform = carla.Transform(carla.Location(x=1.5, z=1.0))
        self.sensor_collision = self.world.spawn_actor(self.collision_bp,collision_transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor_collision)
        self.sensor_collision.listen(lambda data: self.check_collision(data))
       
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, reverse=True))
        
        self.start_time = time.time()

        while self.dash_cam is None:
            time.sleep(0.01)

        return self.dash_cam

    def process_image(self,data):
        raw = np.array(data.raw_data)
        img_trans = raw.reshape((self.img_height, self.img_width, 4))
        img_trans = img_trans[:, :, :3]
        self.dash_cam = img_trans

    def check_collision(self,data):
        # check for collision and act accordingly
        self.collision = True
        print('Collision!!')

    def cleanup(self):
        for actor in self.actor_list:
            if actor is not None:
                actor.destroy()


    def show_cam(self):
        fp_view = self.dash_cam[:,:,:3]
        cv2.imshow("",fp_view)
        cv2.waitKey(1)
