import os, sys, glob
import numpy as np
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

class Carlaenv():

    def __init__():

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)

        # Image height and width
        self.img_height = 600
        self.img_width = 800
        self.inpuut_image = None

        # Once we have a client we can retrieve the world that is currently
        self.world = self.client.get_world()

        # The world contains the list blueprints that we can use for adding new
        blueprint_library = self.world.get_blueprint_library()

        self.vehicle_bp = random.choice(blueprint_library.filter('vehicle.toyota'))

        # The destination will be another spawn point on the map
        self.destination = random.choice(self.world.get_map().get_spawn_points())
        self.start_point = random.choice(self.world.get_map().get_spawn_points())
        self.destination_loc = self.destination.get_forward_vector()
        self.destination_vec = np.array([self.destination_loc[0], self.destination_loc[1]])



    def setup_sensors(self):
        # setup a dashcam sensor
        # Find the blueprint of the sensor.
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute('image_size_x', int(self.img_width))
        self.camera_bp.set_attribute('image_size_y', int(self.img_height))
        self.camera_bp.set_attribute('fov', '120')
        # Set the time in seconds between sensor captures
        self.camera_bp.set_attribute('sensor_tick', '1.0') 
        # move the camera to the dash of the car
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

        self.sensor_cam = self.world.spawn_actor(self.camera_bp, camera_transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor_cam)


    def step(self, action):
        # process action
        speed, steering_angle, braking = action
        # Will have to scale/normalize the actions

        self.vehicle.apply_control(carla.VehicleControl(throttle=speed, steer=steering_angle, braking=braking))
        self.vehicle.
        # get the image (next state)
        self.sensor_cam.listen(lambda data: process_image(data))

        # state is the image
        image_state = self.input_image
        position = self.vehicle.get_location()  # m
        velocity = self.vehicle.get_velocity()  # m/s
        acceleration = self.vehicle.get_acceleration() # m/s^2
        
        pos = np.array([position[0], position[1]])
        
        distance = np.sqrt((self.destination_vec[0]-pos[0])**2 + (self.destination_vec[1]-pos[1])**2)
        # calculate reward
        done = False
        reward, done = self.reward(distance,action)        

        state = (image_state, pos)
        return state, rewaard, done

    def reward(self, distance, action):
        '''The reward signal takes the current state of the agent into account.
        The agent is penalized for bad driving habits (hard braking, high acceleration,
        sharp turns, driving too close to other vehicles)'''

        speed, steering, brake = action
        # check for collision
        self.sensor_collision.listen(lambda data: check_collision(data))
        if self.collision:
            collision = 10
            self.collision = False
        # reward = destination + speed_limit_threshold - break_force - collisions - change streeing angle -
                    # jerk + distance_from_cars_threshold
        reward = -distance - collision

        if 1.0 <= distance <= 3.0:
            reward = 10
            done = True


        return rewaard, done

    def reset(self):

        self.actor_list = []
        # So let's tell the world to spawn the vehicle.
        self.vehicle = self.world.spawn_actor(self.bp, self.start_point)
        self.collision = False
        self.setup_sensors()
        self.sensor_cam.listen(lambda data: self.process_image(data))
        
        # setup the collision sensor
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        collision_transform = self.carla.Transform(self.carla.Location(x=1.5, z=1.0))
        self.sensor_collision = self.world.spawn_actor(self.collision_bp,collision_transform, attach_to=self.vehicle)

        self.actor_list.append(self.sensor_collision)
        self.sensor_collision.listen(lambda data: self.check_collision(data))
        self.vehicle.apply_control(self.carla.VehicleControl(throttle=0.0, steer=0.0, reverse=True))
        time.sleep(2)

    def process_image(self,data):
        raw = np.array(data)
        self.input_image = np.resshape(raw, (self.img_height, self.img_width, 4))
    # will have to reshape data to an image (w x h x 4)

    def check_collision(self,data):
        # check for collision and act accordingly
        self.collision = True
        print('Collision!!!')
        pass


    def drive_loop(vehicle):
        while True:
            # This is where the action is taken (RL action)
            vehicle.apply_control(carla.VehicleControl(throttle=random.random(), steer=random.random(), reverse=False))
            time.sleep(10)
            vehicle.apply_control(carla.VehicleControl(throttle=random.random(),brake=1.0, reverse=False))
            time.sleep(10)
            vehicle.apply_control(carla.VehicleControl(throttle=random.random(), steer=random.random(), reverse=True))
            time.sleep(10)