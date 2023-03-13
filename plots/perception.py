import matplotlib.pyplot as plt


def plot_perception(log, width):
    est_angles = log['est_angles']
    actions = log['actions']

    # Create plot
    fig, axs = plt.subplots(2, 3, num='Perception', figsize=(30, 15))

    # Plot perception
    for j in range(3):
        axs[0, j].set_title(r'$\mu_%d$' % j)
        axs[0, j].set_ylim(-10, 135)
        axs[0, j].xaxis.set_ticklabels([])
        if j == 0:
            axs[0, j].set_ylabel(r'$Angle (\theta)$')
        else:
            axs[0, j].yaxis.set_ticklabels([])
        for i in est_angles[:, :, j]:
            axs[0, j].plot(i, linewidth=width)

    # Plot action
    for j in range(3):
        axs[1, j].set_title(r'$a_%d$' % j)
        axs[1, j].set_xlabel('t')
        axs[1, j].set_ylim(-0.02, 0.02)
        if j == 0:
            axs[1, j].set_ylabel(r'$Velocity$')
        else:
            axs[1, j].yaxis.set_ticklabels([])
        for i in actions[:, :, j]:
            axs[1, j].plot(i, linewidth=width)

    fig.savefig('plots/perception', bbox_inches='tight')
