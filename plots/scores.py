import glob
import re
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import seaborn as sns
import utils
import config as c


def plot_scores():
    ylabels = ['Accuracy (%)', 'Error (px)', 'Time', 'Stability (px)']
    measures = ['acc', 'error', 'time', 'std']
    limits = [(50, 80), (10, 20), (0, 180), (0, 4.0)]
    params = ['alpha', 'k', 'pivt']
    titles = [r'$\alpha$', 'k', r'$\pi_{v_t}$']
    orders = [['BL', '0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6',
              '0.7', '0.8'],
              ['BL', '0.04', '0.06', '0.08', '0.10', '0.12', '0.14', '0.16',
               '0.18', '0.20'],
              ['BL', '1.5e-4', '2e-4', '3e-4', '4e-4', '6e-4', '8e-4',
               '1e-3', '1.2e-3', '1.5e-3']]
    orders_phases = ['immediate', 'dynamic', 'fixed']

    display_precisions(params, titles, measures, ylabels, limits, orders)
    display_belief(titles, measures, orders[0])
    display_phases(measures, ylabels, limits, orders_phases)
    display_model(limits, ['BL', 'full', 'proprioception'])


def display_precisions(params, titles, measures, ylabels,
                       limits, orders):
    fig, axs = plt.subplots(4, 3, num='precisions', figsize=(30, 30))
    display = ['hand', 'hand', 'target']

    for i, p, d, t, o in zip(range(3), params, display, titles, orders):
        score = {'val': [], 'acc': [], 'error': [], 'time': [], 'std': []}

        if d == 'hand':
            ground, est, color = 'target_pos', 'hand_pos', 'Blues_d'
        elif d == 'target':
            ground, est, color = 'target_pos', 'est_target_pos', 'Reds_d'
        else:
            ground, est, color = 'hand_pos', 'est_hand_pos', 'Greens_d'

        for log_name in sorted(glob.glob('simulation/log_%s_*.npz' % p)):
            # Load log
            log = np.load(log_name)
            _, val = re.findall(r'log_(.*)_(.*).npz', log_name)[0]

            data = utils.get_score(log[ground], log[est], mean=False)

            for n in range(len(data[0])):
                score['val'].append(val)
                for m, measure in enumerate(measures):
                    x = data[m][n] if n < len(data[m]) else \
                        np.mean(data[m])
                    score[measure].append(x)

        for j, ylabel, measure, limit in zip(range(4), ylabels,
                                             measures, limits):
            if measure == 'acc':
                sns.barplot(x='val', y=measure, ax=axs[j, i], data=score,
                            palette=color, ci=None, order=o)
            else:
                sns.barplot(x='val', y=measure, ax=axs[j, i], data=score,
                            palette=color, order=o)

            axs[j, i].set_ylim(*limit)
            axs[j, i].tick_params(axis='both', labelsize=14)
            if i == 0:
                axs[j, i].set_ylabel(ylabel)
            else:
                axs[j, i].set_ylabel(None)
        axs[0, i].set_title(t)

    plt.savefig('plots/precisions', bbox_inches='tight')


def display_phases(measures, ylabels, limits, order):
    score = {'phases': [], 'data1': [], 'data2': [], 'acc': [], 'error': [],
             'time': [], 'std': [], 'onset': []}

    fig, axs = plt.subplots(1, 3, num='phases', figsize=(30, 8))

    for log_name in sorted(glob.glob('simulation/log_phases_*.npz')):
        # Load log
        log = np.load(log_name)
        val = re.findall(r'log_phases_(.*).npz', log_name)[0]

        data = [utils.get_score(log['target_pos'], log['hand_pos'], mean=False),
                utils.get_score(log['target_pos'], log['est_target_pos'],
                                mean=False)]

        onset = []
        error = norm(log['target_pos'] - log['hand_pos'], axis=2)
        for e, a in zip(error, log['a']):
            reached = np.where(e < c.reach_dist)
            if reached[0].size > 0:
                start = np.nonzero(a)
                if start[0].size > 0:
                    onset.append(reached[0][0] - start[0][0])

        for i in range(len(data[0][0])):
            score['phases'] += [val, val]
            score['data1'] += ['reach', 'perception']
            score['data2'] += ['from start', 'movement onset']
            for m, measure in enumerate(measures):
                x1 = data[0][m][i] if i < len(data[0][m]) else \
                    np.mean(data[0][m])
                x2 = data[1][m][i] if i < len(data[1][m]) else \
                    np.mean(data[1][m])
                score[measure] += [x1, x2]

            o1 = data[0][2][i] if i < len(data[0][2]) else np.mean(data[0][2])
            o2 = onset[i] if i < len(onset) else np.mean(onset)
            score['onset'] += [o1, o2]

    for j, ylabel, measure, limit in zip(range(2), [ylabels[1], ylabels[3]],
                                         [measures[1], measures[3]],
                                         [limits[1], limits[3]]):
        axs[j].set_xlabel('Modality')
        axs[j].set_ylabel(ylabel)
        axs[j].set_ylim(*limit)

        sns.barplot(x='phases', y=measure, ax=axs[j], hue='data1',
                    data=score, palette=('b', 'r'), order=order)

    axs[2].set_xlabel('Modality')
    axs[2].set_ylabel('Reach time')
    axs[2].set_ylim(0, 180)

    sns.barplot(x='phases', y='onset', ax=axs[2], hue='data2',
                data=score, palette=('g', 'm'), order=order)

    plt.savefig('plots/phases', bbox_inches='tight')
    plt.close()


def display_model(limits, order):
    fig, axs = plt.subplots(1, 3, num='model', figsize=(30, 8))

    score = {'model': [], 'acc_h': [], 'std_h': [], 'std_b': []}

    for log_name in sorted(glob.glob('simulation/log_model_*.npz')):
        # Load log
        log = np.load(log_name)
        val = re.findall(r'log_model_(.*).npz', log_name)[0]

        data = [utils.get_score(log['target_pos'], log['hand_pos'], mean=False),
                utils.get_score(log['hand_pos'], log['est_hand_pos'],
                                mean=False)]

        for i in range(len(data[0][0])):
            score['model'].append(val)
            score['acc_h'].append(data[0][0][i])
            x1 = data[0][3][i] if i < len(data[0][3]) else \
                np.mean(data[0][3])
            x2 = data[1][3][i] if i < len(data[1][3]) else \
                np.mean(data[1][3])
            score['std_h'].append(x1)
            score['std_b'].append(x2)

    measures = ['acc_h', 'std_h', 'std_b']
    ylabels = ['Reach accuracy (%)', 'Reach stability (px)',
               'Belief stability (px)']
    colors = ['Blues_d', 'Blues_d', 'Greens_d']

    for j, ylabel, measure, limit, color in zip(range(3), ylabels, measures,
                                                [limits[0], limits[3],
                                                 limits[3]], colors):
        axs[j].set_xlabel('Model')
        axs[j].set_ylabel(ylabel)
        axs[j].set_ylim(*limit)

        if measure == 'acc':
            sns.barplot(x='model', y=measure, ax=axs[j], data=score,
                        palette=color, ci=None, order=order)
        else:
            sns.barplot(x='model', y=measure, ax=axs[j], data=score,
                        palette=color, ci=None, order=order)

    plt.savefig('plots/model', bbox_inches='tight')
    plt.close()


def display_belief(titles, measures, order):
    fig, axs = plt.subplots(1, 2, num='alpha_belief', figsize=(30, 10))

    score = {'val': [], 'acc': [], 'error': [], 'time': [], 'std': []}

    ground, est, color = 'hand_pos', 'est_hand_pos', 'Greens_d'

    for log_name in sorted(glob.glob('simulation/log_alpha_*.npz')):
        # Load log
        log = np.load(log_name)
        val = re.findall(r'log_alpha_(.*).npz', log_name)[0]

        data = utils.get_score(log[ground], log[est], mean=False)

        for i in range(len(data[0])):
            score['val'].append(val)
            for m, measure in enumerate(measures):
                x = data[m][i] if i < len(data[m]) else \
                    np.mean(data[m])
                score[measure].append(x)

    ylabels = ['Belief error (px)', 'Belief stability (px)']
    for i, ylabel, measure, limit in zip(range(2), ylabels,
                                         [measures[1], measures[3]],
                                         [(0, 5), (0, 3)]):
        sns.barplot(x='val', y=measure, ax=axs[i], data=score,
                    palette=color, order=order)

        axs[i].set_xlabel(titles[0])
        axs[i].set_ylabel(ylabel)
        axs[i].set_ylim(*limit)
        axs[i].tick_params(axis='both', labelsize=24)

        plt.savefig('plots/alpha_belief', bbox_inches='tight')
