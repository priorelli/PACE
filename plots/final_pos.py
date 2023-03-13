import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import config as c
from environment.arm import Arm


def plot_final_pos(log):
    arm = Arm()

    # Create plot
    fig, axs = plt.subplots(1, 2, num='Final positions', figsize=(30, 12))

    for w, cl, var, title in zip(range(2), ['Blues_d', 'Greens_d'],
        ['hand_pos', 'est_hand_pos'],
        ['Hand', 'Hand belief']):
        # Load data
        data = log[var]
        data = data[np.arange(0, len(data), 2)]
        final_points = {'ep': [], 'x': [], 'y': []}

        n = int(len(data) / c.n_reps)
        points = data[:, -1].reshape(n, c.n_reps, 2)
        for t, target in enumerate(points):
            for p in target:
                final_points['ep'].append(t)
                final_points['x'].append(p[0])
                final_points['y'].append(p[1])

        # for e in range(len(data)):
        #     for i in data[e, -200:]:
        #         final_points['ep'].append(e)
        #         final_points['x'].append(i[0])
        #         final_points['y'].append(i[1])

        # Plot targets
        for joint in c.targets:
            pos = arm.kinematics(joint)[-1, :2]
            axs[w].scatter(*pos, marker='o', color='r', s=6000)

        sns.scatterplot(data=final_points, x='x', y='y', palette=cl,
                        hue='ep', s=80, legend=False, ax=axs[w])

        axs[w].set_title(title)
        axs[w].set_xlim(-35, 35)
        axs[w].set_ylim(15, 75)
        axs[w].set_xlabel('x')
        if w == 0:
            axs[w].set_ylabel('y')
        else:
            axs[w].set_ylabel(None)

    plt.savefig('plots/final_points', bbox_inches='tight')
