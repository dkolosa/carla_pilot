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


    def step(self):
        # process action
        self.vehicle.apply_control(carla.VehicleControl(throttle=random.random(), steer=random.random(), reverse=False))

        # get the image
        self.sensor_cam.listen(lambda data: process_image(data))

        # check for collision
        self.sensor_collision.listen(lambda data: check_collision(data))

        # state is the image
        # reward (distnce)
        self.calculate_reward()
        
        done = False
        # check for collision
        if collision:
            reward = -10
        # done (collision??)
        return state, rewaard, done

    def calculate_reward(self):
        
        return rewaard

    def reset(self):

        self.actor_list = []
        # random transform from the list of recommended spawn points of the map.
        transform = random.choice(self.world.get_map().get_spawn_points())

        # So let's tell the world to spawn the vehicle.
        self.vehicle = self.world.spawn_actor(self.bp, transform)

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



    def process_image(data):
        raw = np.array(data)
        self.input_image = np.resshape(raw, (self.img_height, self.img_width, 4))
    # will have to reshape data to an image (w x h x 4)

    def check_collision(data):
        # check for collision and act accordingly
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