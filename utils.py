import numpy as np
import torch
from torch.utils import data
import argparse
import sys
import config as c


# Define dataset class
class Dataset(data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


# Split dataset in train/test set and in batches
def split_dataset(x, y, percent):
    length = int(len(x) * percent)
    dataset = Dataset(x, y)
    train_set, test_set = data.random_split(dataset, (length, len(x) - length))
    train_gen = data.DataLoader(train_set, batch_size=c.n_batch, shuffle=True)
    test_gen = data.DataLoader(test_set, batch_size=c.n_batch)

    return train_gen, test_gen


# Compute Kullback-Leibler divergence
def kl_divergence(p_m, p_v, q_m, q_v):
    return torch.mean(0.5 * torch.sum(torch.log(p_v) - q_v +
                                      (q_v.exp() + (q_m - p_m) ** 2) / p_v - 1,
                                      dim=1), dim=0)


# Add Gaussian noise to array
def add_gaussian_noise(array, noise):
    sigma = noise ** 0.5
    return array + np.random.normal(0, sigma, np.shape(array))


# Normalize data
def normalize(x, limits):
    return (x - limits[0]) / (limits[1] - limits[0])


# Denormalize data
def denormalize(x, limits):
    return x * (limits[1] - limits[0]) + limits[0]


# Reverse dictionary
def reverse_dict(dict_):
    reversed_dict = {}

    for key, value in dict_.items():
        reversed_dict.setdefault(value, []).append(key)

    return reversed_dict


# Parse arguments for simulation
def get_sim_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--generate-data',
                        action='store_true', help='Generate dataset')
    parser.add_argument('-t', '--test_network',
                        action='store_true', help='Test network')
    parser.add_argument('-m', '--manual-control',
                        action='store_true', help='Start manual control')
    parser.add_argument('-s', '--simulation',
                        action='store_true', help='Start simulation')
    parser.add_argument('-a', '--ask-params',
                        action='store_true', help='Ask parameters')

    args = parser.parse_args()
    return args


# Parse arguments for plots
def get_plot_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dynamics',
                        action='store_true', help='Plot dynamics')
    parser.add_argument('-p', '--perception',
                        action='store_true', help='Plot perception')
    parser.add_argument('-a', '--attractors',
                        action='store_true', help='Plot attractors')
    parser.add_argument('-g', '--gradients',
                        action='store_true', help='Plot gradients')
    parser.add_argument('-f', '--final',
                        action='store_true', help='Plot final positions')
    parser.add_argument('-s', '--scores',
                        action='store_true', help='Plot scores')
    parser.add_argument('-v', '--video',
                        action='store_true', help='Record video')

    args = parser.parse_args()
    return args


# Compute score
def get_score(ground, est, mean=True):
    error = np.linalg.norm(ground - est, axis=2)
    final_error = error[:, -1]
    acc = (final_error < c.reach_dist) * 100

    time, std = [], []
    for ep in error:
        reached = np.where(ep < c.reach_dist)
        if reached[0].size > 0:
            time.append(reached[0][0])
            std.append(np.std(ep[reached[0][0]:]))

    if mean:
        return np.mean(acc), np.mean(final_error), \
               np.mean(time), np.mean(std)
    else:
        return acc, final_error, time, std


# Print score
def print_score(log, time):
    score = np.array([get_score(log.target_pos, log.hand_pos),
                      get_score(log.hand_pos, log.est_hand_pos),
                      get_score(log.target_pos, log.est_target_pos)])

    print('\n' + '=' * 30)
    print('\t\tHand\t\tBelief\t\tTarget')
    for m, measure in enumerate(('Acc', 'Error', 'Time', 'Std')):
        print('{:s}\t\t{:.2f}\t\t{:.2f}\t\t{:.2f}'.format(
            measure, *score.T[m]))
    print('Time elapsed: {:.2f}'.format(time))


# Print simulation info
def print_info(trial, success, step, dt):
    dt = 0.001 if dt == 0 else dt
    sys.stdout.write('\rTrial: {:d}({:d})/{:d} \t '
                     'Step: {:3d}/{:d} \t FPS: {:3.0f}'
                     .format(trial, int(success), c.n_trials,
                             step, c.n_steps, 1 / dt))
    sys.stdout.flush()
