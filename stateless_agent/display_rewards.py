import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import draw, show
import time

def display_rewards(file_names):
    assert isinstance(file_names, list)
    f, axs = plt.subplots(1,2,sharey=True)

    for i in range(len(file_names)):
        file_name = file_names[i]
        ax = axs[i]
        rewards = list(np.load(file_name))

        rewards = np.array(rewards)
        x = np.linspace(0, len(rewards)-1, len(rewards)).repeat(rewards.shape[1])

        mean = np.mean(rewards, axis=1)
        mx = np.max(rewards, axis=1)
        mn = np.min(rewards, axis=1)

        ax.scatter(x, rewards.flatten(), s=8)
        ax.plot(mean, c='g', label='Averages')
        ax.plot(mn, c='r', label='Minimum')
        ax.plot(mx, c='y', label='Maximums')

        ax.set_title(file_name)
        ax.legend(loc='upper left')

        ax.set_xlabel('Generations', fontsize=10)
        ax.set_xticks(np.arange(min(x), max(x)+1, 10))

        ax.set_ylabel('Reward', fontsize=10)


    show(block=False)
        # plt.pause(60)
        # plt.close(fig)

if __name__ == '__main__':
    file_names = ['rewards_continuous.npy', 'rewards_gumbel.npy']
    display_rewards(file_names)

    plt.show()
