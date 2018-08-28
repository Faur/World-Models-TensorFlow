import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

from gym.envs.box2d import CarRacing
from gym.envs.box2d.car_dynamics import Car

import generate_VAE_data

# from train_VAE import load_vae
# from train_VAE import _EMBEDDING_SIZE, _EXP_NAME
from train_Gumbel_VAE import load_vae
from train_Gumbel_VAE import _EMBEDDING_SIZE, _EXP_NAME

_TIME_STEPS = 300
_RENDER = True

def main(args):
    plt.ion()
    controller = args.controller

    sess, model = load_vae()

    # while True:
    generate_VAE_data._BATCH_SIZE = 1
    # generate_VAE_data._BATCH_SIZE = 1
    data = generate_VAE_data.simulate_batch(0)
    data = np.array(data).astype('float32') #/ 255.
    data = resize(data, (data.shape[0], 64, 64, 3), anti_aliasing=True, mode='reflect')

    title = 'test_seq'
    model.show_pred(title, data=data)

    # pred = model.model.predict(data, verbose=0)
    # logits, pre_gumbel_softmax, gumbel, hard_sample = model.encoder([data])
    # model.show_pred(title, data=data, logits=logits, pre_gumbel_softmax=pre_gumbel_softmax,
    #               gumbel=gumbel, hard_sample=hard_sample, pred=pred)

    sess.close()  # we don't need this


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Human interface')

    parser.add_argument('--show', type=str, default='both', choices=['vae', 'sim', 'both'], help='Show the output of the VAE?')
    parser.add_argument('--controller', type=str, default='random', choices=['human', 'agent', 'random'],
                        help="How should actions be selected? ['human', 'agent', 'random']")
    args = parser.parse_args()
    print(args, '\n')

    main(args)
