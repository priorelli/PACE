import numpy as np
import matplotlib.pyplot as plt
import config as c


def moving_average(x):
    return np.convolve(x, np.ones(30), 'valid') / 30


def plot_attractors(log, width):
    trial = 1

    # Load variables
    mu = log['mu']
    mu_dot = log['mu_dot']
    actions_dot = log['actions_dot']
    df_dmu = [mu_dot[:, :, 0] - mu[:, :, 1], mu_dot[:, :, 1]]

    mu_lims = [(-0.005, 0.005), (-0.0025, 0.0025)]
    mud_lims = (-0.012, 0.015)
    mut_lims = (-0.015, 0.015)
    actions_lims = (-0.002, 0.002)

    # Create plots
    fig1, axs1 = plt.subplots(3, 2, num='df', figsize=(30, 30))
    fig2, axs2 = plt.subplots(1, 2, num='mu_dot', figsize=(30, 10))

    for j in range(2):
        # Plot free energy over arm belief
        for l, label, lim in zip(range(2), ['mu_{a', 'mu\'_{a'], mu_lims):
            axs1[l, j].set_title(r'$\frac{\partial F}{\partial \%s_%d}}$' %
                                 (label, j + 1), fontsize=50)
            if j == 0:
                axs1[l, j].set_ylabel('Force')
            else:
                axs1[l, j].yaxis.set_ticklabels([])
            axs1[l, j].set_ylim(lim)
            axs1[l, j].xaxis.set_ticklabels([])

            for i in df_dmu[l][:, :, j + 1]:
                axs1[l, j].plot(moving_average(i), linewidth=width)
            axs1[l, j].yaxis.set_major_locator(plt.MaxNLocator(4))
            axs1[l, j].plot(np.full(c.n_steps, 0), '--', color='b',
                            linewidth=width - 2)
            axs1[l, j].axvline(int(c.n_steps / c.phases_ratio) - 30,
                               linewidth=width - 2)

        # Plot free energy over target belief
        axs1[2, j].set_title(r'$\frac{\partial F}{\partial \mu_{t_%d}}$' %
                             (j + 1), fontsize=50)
        if j == 0:
            axs1[2, j].set_ylabel('Force')
        else:
            axs1[2, j].yaxis.set_ticklabels([])
        axs1[2, j].set_xlabel('t')
        axs1[2, j].set_ylim(mut_lims)

        for i in df_dmu[0][:, :, j + 4]:
            axs1[2, j].plot(moving_average(i), linewidth=width)
        axs1[2, j].yaxis.set_major_locator(plt.MaxNLocator(4))
        axs1[2, j].plot(np.full(c.n_steps, 0), '--', color='b',
                        linewidth=width - 2)
        axs1[2, j].axvline(int(c.n_steps / c.phases_ratio) - 30,
                           linewidth=width - 2)

        # Plot free energy over action
        # axs1[2, j].set_title(r'$\frac{\partial F}{\partial a_%d}$' %
        #                      (j + 1), fontsize=40)
        # if j == 0:
        #     axs1[2, j].set_ylabel('Force')
        # else:
        #     axs1[2, j].yaxis.set_ticklabels([])
        # axs1[2, j].set_xlabel('t')
        # axs1[2, j].set_ylim(actions_lims)
        #
        # for i in actions_dot[:, :, j + 1]:
        #     axs1[2, j].plot(moving_average(i), linewidth=width)
        # axs1[2, j].yaxis.set_major_locator(plt.MaxNLocator(4))
        # axs1[2, j].plot(np.full(c.n_steps, 0), '--', color='b',
        #                 linewidth=width - 2)

        # Plot free energy over frame of reference
        axs2[j].set_title(r'$\mu_%d$' % (j + 1))
        if j == 0:
            axs2[j].set_ylabel('Force')
        else:
            axs2[j].yaxis.set_ticklabels([])
        axs2[j].set_xlabel('t')
        axs2[j].set_ylim(mud_lims)
        axs2[j].plot(np.full(c.n_steps, 0), '--', color='b',
                     linewidth=width - 2)

        for label, traj in zip([r'$\frac{\partial F}{\partial \mu}$',
                               r'$\dot{\mu}$', r"$\mu'$"],
                               [df_dmu[0], mu_dot[:, :, 0], mu[:, :, 1]]):
            axs2[j].plot(moving_average(traj[trial, :, j + 1]),
                         label=label, linewidth=width)
        axs2[j].legend(prop={'size': 40}, loc='upper right')
        axs2[j].yaxis.set_major_locator(plt.MaxNLocator(4))
        axs2[j].axvline(int(c.n_steps / c.phases_ratio) - 30,
                        linewidth=width - 2)

    fig1.savefig('plots/df', bbox_inches='tight')
    fig2.savefig('plots/mu_dot', bbox_inches='tight')
