import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy.linalg import norm
import config as c


def plot_dynamics(log, width):
    # Load variables
    target_pos = log['target_pos']
    est_target_pos = log['est_target_pos']

    hand_pos = log['hand_pos']
    est_hand_pos = log['est_hand_pos']

    e_hand = norm(target_pos - hand_pos, axis=2)
    e_belief = norm(hand_pos - est_hand_pos, axis=2)
    e_target = norm(target_pos - est_target_pos, axis=2)

    # Create plot
    fig, axs = plt.subplots(1, 2, num='dynamics', figsize=(30, 10))

    for w, error, color, title, lims in zip(range(2), [e_hand, e_belief],
        ['Blues_d', 'Greens_d'], ['Hand', 'Hand belief'], [(0, 50), (0, 10)]):
    # for w, error, color, title, lims in zip(range(2), [e_hand, e_target],
    #     ['Blues_d', 'Reds_d'], ['Hand', 'Target'], [(0, 50), (0, 50)]):
        # Compute error
        if c.task == 'all':
            error = error[np.arange(0, len(error), 2)]

        lines = {'ep': [], 'x': [], 'y': []}
        for e, line in enumerate(error):
            for i, val in enumerate(line):
                lines['ep'].append(e)
                lines['x'].append(i)
                lines['y'].append(val)

        # Plot error
        if c.task == 'single':
            cl = 'Blue' if color == 'Blues_d' else 'Red' \
                if color == 'Reds_d' else 'Green'
            axs[w].plot(np.full(c.n_steps, c.reach_dist), '--')
            sns.lineplot(x='x', y='y', ax=axs[w], data=lines, color=cl,
                         linewidth=width, legend=False, ci='sd')
        else:
            axs[w].plot(np.full(c.n_steps, c.reach_dist), '--')
            sns.lineplot(x='x', y='y', ax=axs[w], data=lines, palette=color,
                         linewidth=width, hue='ep', legend=False)

        axs[w].set_title(title)
        axs[w].set_ylim(*lims)
        axs[w].set_xlabel('t')
        if w == 0:
            axs[w].set_ylabel('L2 Norm (px)')
        else:
            axs[w].set_ylabel(None)

    fig.savefig('plots/dynamics', bbox_inches='tight')
