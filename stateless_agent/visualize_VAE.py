import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

# from gym.envs.box2d import CarRacing
# from gym.envs.box2d.car_dynamics import Car

import generate_VAE_data

import train_VAE 
import train_Gumbel_VAE

def generate_data():
    generate_VAE_data._BATCH_SIZE = 1
    data96 = generate_VAE_data.simulate_batch(0, False)
    data96 = np.array(data96).astype('float32')
    data64 = resize(data96, (data96.shape[0], 64, 64, 3), anti_aliasing=True, mode='reflect')
    return data96, data64

def process_data(data, title, gumbel=True):
    plt.ion()
    if gumbel:
        sess, model = train_Gumbel_VAE.load_vae()
    else:
        sess, model = train_VAE.load_vae()

    pred = model.model.predict(data, verbose=0)
    logits, pre_gumbel_softmax, gumbel, hard_sample = model.encoder([data])

    max = len(data)
    for n in range(max):
        fig = plt.figure(1)
        print('\r' + title+'-'+str(n) + ' - Plotting vae pred: {}/{}'.format(n, max), end='')

        model.make_image(fig, title + ' i' + str(n), data[n], logits[n],
                        pre_gumbel_softmax[n],
                        gumbel[n], hard_sample[n], pred[n])
        plt.show()

        # plt.savefig('./videos/CarRacing-'+title+'-'+str(n)+'.png', cmap='hot', bbox_inches='tight')
        # plt.close()
    sess.close()  # we don't need this


def main(args):
    data = generate_data()
    
    # controller = args.controller
    title = 'test_seq'
    process_data(data, title)


    model.show_pred(title, data=data, logits=logits, pre_gumbel_softmax=pre_gumbel_softmax,
                  gumbel=gumbel, hard_sample=hard_sample, pred=pred)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Human interface')

    parser.add_argument('--show', type=str, default='both', choices=['vae', 'sim', 'both'], help='Show the output of the VAE?')
    # parser.add_argument('--controller', type=str, default='random', choices=['human', 'agent', 'random'],
    #                     help="How should actions be selected? ['human', 'agent', 'random']")
    args = parser.parse_args()
    print(args, '\n')

    main(args)
