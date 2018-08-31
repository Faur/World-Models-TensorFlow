"""
Game is solved when agent consistently gets 900+ points. Track is random every episode.
"""

import numpy as np
import time, tqdm
import gym
from gym.envs.box2d.car_dynamics import Car
from gym.envs.box2d import CarRacing

import cma
import multiprocessing as mp
import datetime

from train_VAE import load_vae
from train_VAE import _EMBEDDING_SIZE, _EXP_NAME
# from train_Gumbel_VAE import load_vae
# from train_Gumbel_VAE import _EMBEDDING_SIZE, _EXP_NAME

gym.logger.set_level(40)

_IS_TEST = False
# _EMBEDDING_SIZE = 32 # TODO Handle this!
_NUM_PREDICTIONS = 2  # TODO: This is cheating!
_NUM_ACTIONS = 3
_NUM_PARAMS = _NUM_PREDICTIONS * _EMBEDDING_SIZE + _NUM_PREDICTIONS

if _IS_TEST: print("!!! _IS_TEST is True !!!")
print('EXP_NAME', _EXP_NAME)


def get_weights_bias(params):
    assert params is not None, 'params is None!'
    weights = params[:_NUM_PARAMS - _NUM_PREDICTIONS]
    bias = params[-_NUM_PREDICTIONS:]
    weights = np.reshape(weights, [_EMBEDDING_SIZE, _NUM_PREDICTIONS])
    return weights, bias

def decide_action(sess, embedding, params):
    # embedding = sess.run(network.z, feed_dict={network.image: observation[None, :,  :,  :]})
    weights, bias = get_weights_bias(params)

    action = np.zeros(_NUM_ACTIONS)
    prediction = np.matmul(np.reshape(embedding, -1), weights) + bias
    prediction = np.tanh(prediction)

    action[0] = prediction[0]
    if prediction[1] < 0:
        action[1] = np.abs(prediction[1])
        action[2] = 0
    else:
        action[2] = prediction[1]
        action[1] = 0

    return action


def play(params, render=True, verbose=False, save_visualization=False, max_len=999):
    print('Agent train run begun', datetime.datetime.now())

    sess, network = load_vae()
    env = CarRacing()

    # _NUM_TRIALS = 12
    _NUM_TRIALS = 8

    agent_reward = 0
    for trial in range(_NUM_TRIALS):
        observation = env.reset()
        observation = network.normalize_observation(observation)
        # Little hack to make the Car start at random positions in the race-track
        np.random.seed(int(str(time.time()*1000000)[10:13]))
        position = np.random.randint(len(env.track))
        env.car = Car(env.world, *env.track[position][1:4])

        total_reward = 0.0
        steps = 0
        observations = [observation]
        while True:
            if render:
                env.render()
            observation = network.normalize_observation(observation)
            observations.append(observation)

            embedding = network.get_embedding(sess, observation)
            action = decide_action(sess, embedding, params)
            observation, r, done, info = env.step(action)
            total_reward += r
            # NB: done is not True after 1000 steps when using the hack above for
            #       random init of position
            if verbose and (steps % 200 == 0 or steps == 999):
                print("\naction " + str(["{:+0.2f}".format(x) for x in action]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))

            steps += 1
            if steps == max_len:
                break
            # if total_reward < -50:
            #     break
            if _IS_TEST and steps > 10:
                break

        total_reward = np.maximum(-100, total_reward)
        agent_reward += total_reward
        if save_visualization:
            title = 'train_agent_r{:.2f}'.format(agent_reward)
            print('Saving trajectory:', title)
            network.show_pred(title, np.concatenate(observations, 0))
            break

    sess.close()
    env.close()
    return - (agent_reward / _NUM_TRIALS)


def train():
    multi_thread = True
    print('multi_thread:', multi_thread )

    # popsize = 16
    popsize = 12
    if _IS_TEST:
        popsize = 2
    num_parallel = mp.cpu_count() - 1
    # num_parallel = 4

    print('popsize', popsize)
    print('num_parallel', num_parallel)

    try:
        if _IS_TEST: raise Exception  # skip!
        # TODO: Make sure and test that this actually makes sense!
        prev_model = np.load('best_params_' + _EXP_NAME + '.npy')
        es = cma.CMAEvolutionStrategy(prev_model, 0.1, {'popsize': popsize})
        rewards_through_gens = np.load('rewards_'+_EXP_NAME+'.npy')
        rewards_through_gens = list(rewards_through_gens)
        print('Model loaded')
    except:
        es = cma.CMAEvolutionStrategy(_NUM_PARAMS * [0], 0.1, {'popsize': popsize})
        rewards_through_gens = []
        print('No model to load / loading failed')

    print("\n**************")
    print('Begin training!')
    print('EXP_NAME:  ', _EXP_NAME, ' _NUM_PARAMS', _NUM_PARAMS)
    print("Generation: {}".format(len(rewards_through_gens)))
    print(datetime.datetime.now())
    print("**************\n")

    try:
        while not es.stop():
            solutions = es.ask()
            if multi_thread:
                with mp.Pool(num_parallel) as p:
                    rewards = list(tqdm.tqdm(p.imap(play, list(solutions)), total=len(solutions)))
            else:
                print('NOT MULTI THREADING')
                play(solutions[0], render=True, verbose=False, save_visualization=True)
                es = None
                break

            print("\n**************")
            es.tell(solutions, rewards)
            rewards = np.array(rewards) * (-1.)  # cma is a minimizer, so we negate the reward twice
            rewards_through_gens.append(rewards)
            print('EXP_NAME:  ', _EXP_NAME)
            print("Generation: {}".format(len(rewards_through_gens)))
            print("Min reward: {:.3f}\nMax reward: {:.3f}".format(np.min(rewards), np.max(rewards)))
            print("Avg reward: {:.3f}".format(np.mean(rewards)))
            print("Agent characteristics: params {}, min {:.3f} mean {:.3f}, max {:.3f}, std {:.3f}".format(
                    _NUM_PARAMS, np.min(es.best.get()[0]), np.mean(es.best.get()[0]),
                     np.max(es.best.get()[0]), np.std(es.best.get()[0])))
            print(datetime.datetime.now())
            print("**************\n")

            if len(rewards_through_gens) % 10 == 0:
                play(es.best.get()[0], render=True, verbose=False, save_visualization=True, max_len=200)
            if not _IS_TEST:
                np.save('rewards_'+_EXP_NAME, rewards_through_gens)
                np.save('best_params_'+_EXP_NAME, es.best.get()[0])

    except (KeyboardInterrupt, SystemExit):
        print("Manual Interrupt")
    # except Exception as e:
    #     print("Exception: {}".format(e))
    return es

def main():
    es = train()

    if es is not None:
        print('Training complete')
        if not _IS_TEST:
            np.save('best_params_'+_EXP_NAME, es.best.get()[0])
        play(es.best.get()[0], render=True, verbose=False, save_visualization=True)

        input("Press enter to play... ")
        RENDER = True
        score = play(es.best.get()[0], render=RENDER, verbose=True)
        print("Final Score: {}".format(-score))
    print('Done')


if __name__ == '__main__':
    main()
