import numpy as np
import pygame
import cv2

import TDDDPG
import carla_env

# Based on the carla tutorial script tutorial.py

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

if "__name__" == "main":

    model_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(os.path.join(model_dir, str(datetime.date.today()) + '-' + 'Carla'), exist_ok=True)
    save_dir = os.path.join(model_dir, str(datetime.date.today()) + '-' + 'Carla')

    env = gym.make(ENV)
    iter_per_episode = 200
    # the state is an input image and current speed (+-1)
    n_state = env.observation_space.shape[0]
    # the acions are the steering angle and the speed (-1, +1)
    n_action = 2
    action_bound = 40   # the max mph limit

    env.seed(1234)
    np.random.seed(1234)

    num_episodes = 1001
    PER = True

    batch_size = 64
    
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
    if load_models:
        load_model(PER, agent, batch_size, env, ep, n_action, n_state)

    noise_decay = 1.0

    for i in range(num_episodes):
        s = env.reset()
        sum_reward = 0
        agent.sum_q = 0
        agent.actor_loss = 0
        agent.critic_loss = 0
        j = 0

        while True:
            # env.render()

            a = np.clip(agent.actor(tf.convert_to_tensor([s], dtype=tf.float32))[0] + actor_noise()*noise_decay, a_max=action_bound,
                        a_min=-action_bound)
            s1, r, done, _ = env.step(a)
            # Store in replay memory
            if PER:
                error = 1 # D_i = max D
                agent.memory.add(error, (
                np.reshape(s, (n_state,)), np.reshape(a, (n_action,)), r, np.reshape(s1, (n_state,)), done))
            else:
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


def load_model(PER, agent, batch_size, env, ep, n_action, n_state):
    for i in range(batch_size + 1):
        s = env.reset()
        a = agent.actor(tf.convert_to_tensor([s], dtype=tf.float32))[0]
        s1, r, done, _ = env.step(a)
        # Store in replay memory
        if PER:
            error = abs(r + ep)  # D_i = max D
            agent.memory.add(error, (
                np.reshape(s, (n_state,)), np.reshape(a, (n_action,)), r, np.reshape(s1, (n_state,)), done))
        else:
            agent.memory.add(
                (np.reshape(s, (n_state,)), np.reshape(a, (n_action,)), r, np.reshape(s1, (n_state,)), done))
    agent.train()
    agent.load_model()


def process_image(data):
    raw = np.array(data)
    # will have to reshape data to an image (w x h x 4)

def check_collision(data):
    # check for collision and act accordingly
    pass

def game_loop(args):
    # This sets up the environmnet, have to loop through the environment and update
    actor_list = []

    try:
        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)

        # Once we have a client we can retrieve the world that is currently
        # running.
        world = client.get_world()

        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        blueprint_library = world.get_blueprint_library()

        # Now let's filter all the blueprints of type 'vehicle' and choose one
        # at random.
        bp = random.choice(blueprint_library.filter('vehicle'))

        # A blueprint contains the list of attributes that define a vehicle's
        # instance, we can read them and modify some of them. For instance,
        # let's randomize its color.
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        # Now we need to give an initial transform to the vehicle. We choose a
        # random transform from the list of recommended spawn points of the map.
        transform = random.choice(world.get_map().get_spawn_points())

        # So let's tell the world to spawn the vehicle.
        vehicle = world.spawn_actor(bp, transform)

        # setup a dashcam sensor
        # Find the blueprint of the sensor.
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        camera_bp.set_attribute('image_size_x', '1920')
        camera_bp.set_attribute('image_size_y', '1080')
        camera_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        camera_bp.set_attribute('sensor_tick', '1.0') 
        # move the camera to the dash of the car
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        sensor_cam = carla.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(sensor_cam)
        
        # setup the collision sensor
        collision_bp = world.get_blueprint_library().find('sensor.other.collision')
        sensor_collision = carla.spawn_actor(collision_bp,camera_transform, attach_to=vehicle)

        actor_list.append(sensor_collision)

        #get the image
        sensor_cam.listen(lambda data: process_image(data))

        #check for collision
        sensor_collision.listen(lambda data: check_collision(data))

        # get the velocity as well
        velocity = vehicle.get_velocity()

        # you can also set the velocity (action)
        # vehicle.set_velocity()

        # It is important to note that the actors we create won't be destroyed
        # unless we call their "destroy" function. If we fail to call "destroy"
        # they will stay in the simulation even after we quit the Python script.
        # For that reason, we are storing all the actors we create so we can
        # destroy them afterwards.
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        # Let's put the vehicle to drive around.
        vehicle.set_autopilot(False)

        # This is where the action is taken (RL action)
        vehicle.apply_control(carla.VehicleControl(throttle=1.0))

        # But the city now is probably quite empty, let's add a few more
        # vehicles.
        transform.location += carla.Location(x=40, y=-3.2)
        transform.rotation.yaw = -180.0
        for _ in range(0, 10):
            transform.location.x += 8.0

            bp = random.choice(blueprint_library.filter('vehicle'))

            # This time we are using try_spawn_actor. If the spot is already
            # occupied by another object, the function will return None.
            npc = world.try_spawn_actor(bp, transform)
            if npc is not None:
                actor_list.append(npc)
                npc.set_autopilot(True)
                print('created %s' % npc.type_id)

        time.sleep(5)

    finally:
        print('destroying actors')
        camera.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')
