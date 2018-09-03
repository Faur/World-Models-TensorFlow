import numpy as np
import multiprocessing as mp
from skimage.transform import resize
import gym
from gym.envs.box2d.car_dynamics import Car
from gym.envs.box2d import CarRacing

_BATCH_SIZE = 16
_NUM_BATCHES = 16
_TIME_STEPS = 300
_RENDER = True


def generate_action(prev_action):
    if np.random.randint(3) % 3:
        return prev_action

    index = np.random.randn(3)
    # Favor acceleration over the others:
    index[1] = np.abs(index[1])
    index = np.argmax(index)
    mask = np.zeros(3)
    mask[index] = 1

    action = np.random.randn(3)
    action = np.tanh(action)
    action[1] = (action[1] + 1) / 2
    action[2] = (action[2] + 1) / 2

    return action*mask


def normalize_observation(observation, output_4d=True, reduce_size=True):
    # TODO: Probably reshape
    obs = observation
    if len(observation.shape) == 3:
        if reduce_size:
            obs = resize(observation, (64, 64, 3), anti_aliasing=True, mode='reflect')
        if output_4d:
            obs = np.expand_dims(obs, axis=0)
    elif len(observation.shape) == 4:
        if reduce_size:
            obs = resize(observation, (observation.shape[0], 64, 64, 3),
                         anti_aliasing=True, mode='reflect')

    if np.max(obs) > 1.0:  # TODO: This is shit
        obs = obs / 255.

    return obs.astype('float32')


def simulate_batch(batch_num, save=True, time_steps=None, reduce_size=True):
    env = CarRacing()

    if time_steps is None:
        time_steps = _TIME_STEPS

    obs_data = []
    action_data = []
    action = env.action_space.sample()
    for i_episode in range(_BATCH_SIZE):
        observation = env.reset()
        # Little hack to make the Car start at random positions in the race-track
        position = np.random.randint(len(env.track))
        env.car = Car(env.world, *env.track[position][1:4])
        observation = normalize_observation(observation, output_4d=False, reduce_size=reduce_size)
        obs_data.append(observation)

        for _ in range(time_steps):
            if _RENDER:
                env.render()

            action = generate_action(action)

            observation, reward, done, info = env.step(action)
            observation = normalize_observation(observation, output_4d=False, reduce_size=reduce_size)

            obs_data.append(observation)

    if save:
        print("Saving dataset for batch {:03d}".format(batch_num))
        np.save('../data/obs_data_VAE_{:03d}'.format(batch_num), obs_data)
    
    env.close()
    return obs_data


def main():
    print("Generating data for env CarRacing-v0")
    with mp.Pool(mp.cpu_count()) as p:
        p.map(simulate_batch, range(_NUM_BATCHES))

    data_list = []
    for batch_num in range(0, _NUM_BATCHES - 1):
        batch_to_load = '../data/obs_data_VAE_{:03d}.npy'.format(batch_num)
        try:
            data = np.load(batch_to_load)
            # data = resize(data, (len(data), 64, 64, 3), anti_aliasing=True).astype(np.float32)
            print('\tFound batch at {}. Data size = {} episodes'.format(
                batch_to_load, len(data)))
            data_list.append(data)
        except:
            print('\tUnable to load:', batch_to_load)
    data = np.concatenate(data_list, axis=0)
    np.save('data_combined.npy', data)
    print('data_combined.npy saved.')

    batch_to_load = '../data/obs_data_VAE_{:03d}.npy'.format(_NUM_BATCHES - 1)
    test_data = np.load(batch_to_load)
    # data = resize(data, (len(data), 64, 64, 3), anti_aliasing=True).astype(np.float32)
    print('\tFound batch at {}. Data size = {} episodes'.format(
        batch_to_load, len(test_data)))
    np.save('test_data.npy', test_data)
    print('test_data.npy saved.')


if __name__ == "__main__":
    main()
