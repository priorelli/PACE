import numpy as np
import time
from network.vae import VAE


def main():
    # Load dataset
    dataset = np.load('network/dataset.npz')
    x, y = dataset['x'], dataset['y']

    # Initialize network
    network = VAE()

    # Train network
    start = time.time()
    network.train_net(network, x, y)

    print('\nTime elapsed: {:.2f} seconds'.format(time.time() - start))

    # Save network
    network.save()


if __name__ == '__main__':
    main()
