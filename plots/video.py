import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from pylab import tight_layout
import time
import sys
import config as c
import matplotlib.animation as animation
from environment.arm import Arm


def record_video(log, width):
    # Initialize arm
    arm = Arm()

    # Load variables
    n_t = log['angles'].shape[0] * log['angles'].shape[1]
    angles = log['angles'].reshape(n_t, c.n_joints)
    est_angles = log['est_angles'].reshape(n_t, c.n_joints)

    target_pos = log['target_pos'].reshape(n_t, 2)
    est_target_pos = log['est_target_pos'].reshape(n_t, 2)

    actions = log['actions'].reshape(n_t, c.n_joints)
    mu_dot = log['mu_dot'].reshape(n_t, c.n_orders, c.n_joints * 3)

    mode = log['mode'].reshape(n_t)
    int_errors = log['int_errors'].reshape(n_t, 2, c.n_joints * 3)

    arm_pos = np.zeros((n_t, c.n_joints + 1, 2))
    est_arm_pos = np.zeros((n_t, c.n_joints + 1, 2))
    for i in range(n_t):
        arm_pos[i] = arm.kinematics(angles[i])[:, :2]
        est_arm_pos[i] = arm.kinematics(est_angles[i, :c.n_joints])[:, :2]
    error_h = norm(target_pos - arm_pos[:, -1], axis=1)
    error_t = norm(target_pos - est_target_pos, axis=1)

    # Create plot
    fig, axs = plt.subplots(4, figsize=(20, 30),
                            gridspec_kw={'height_ratios': [4, 1, 1, 1]})
    xlims = [(-40, 85), (0, c.n_steps), (0, c.n_steps), (0, c.n_steps)]
    ylims = [(-15, 65), (-0.01, 0.08), (-0.01, 0.08), (-0.01, 0.06)]
    # ylims = [(-15, 65), (20, 100), (20, 100),  (-0.02, 0.02)]

    def animate(n):
        if (n + 1) % 10 == 0:
            sys.stdout.write('\rEpisode: {:d} \tIter: {:d}'
                             .format(int(n / c.n_steps) + 1,
                                     (n % c.n_steps) + 1))
            sys.stdout.flush()

        # Clear plot
        for w, xlim, ylim in zip(range(4), xlims, ylims):
            axs[w].clear()
            axs[w].get_xaxis().set_visible(False)
            axs[w].get_yaxis().set_visible(False)
            axs[w].set_xlim(xlim)
            axs[w].set_ylim(ylim)
        tight_layout()

        # Draw arm
        for j in range(c.n_joints):
            axs[0].plot(*est_arm_pos[n, j: j + 2].T,
                        linewidth=arm.size[j, 1] * 8, color='g', zorder=1)
            axs[0].plot(*arm_pos[n, j: j + 2].T,
                        linewidth=arm.size[j, 1] * 8, color='b', zorder=1)

        # Draw target
        trial = int(n / c.n_steps)
        axs[0].scatter(*est_target_pos[n], color='m', s=8000, zorder=0)
        if c.task != 'all' or c.task == 'all' and trial % 2 == 0:
            axs[0].scatter(*target_pos[n], color='r', s=8000, zorder=0)

        # Draw belief
        # for j, cl in zip([1, 2, 4, 5], ['blue', 'skyblue', 'darkred', 'red']):
        #     axs[1].plot(mu_dot[n - (n % c.n_steps): n + 1, 0, j + 1],
        #                 label=r'$\dot{\mu}_%d$' % (j + 1),
        #                 linewidth=width + 2, color=cl)
        axs[1].plot(norm(mu_dot[n - (n % c.n_steps): n + 1, 0, :3], axis=1),
                    label=r'$\dot{\mu}_a$', linewidth=width + 2, color='green')
        axs[1].plot(norm(mu_dot[n - (n % c.n_steps): n + 1, 0, 3:6], axis=1),
                    label=r'$\dot{\mu}_t$', linewidth=width + 2, color='red')

        # Draw action
        # for j, cl in enumerate(['darkorange', 'gold']):
        #     axs[2].plot(a[n - (n % c.n_steps): n + 1, j + 1], color=cl,
        #                 label=r'$a_%d$' % (j + 1), linewidth=width + 2)
        axs[2].plot(norm(actions[n - (n % c.n_steps): n + 1], axis=1),
                    label=r'$a$', linewidth=width + 2, color='blue')

        # Draw attractors
        axs[3].plot(norm(int_errors[n - (n % c.n_steps): n + 1, 0], axis=1),
                    label=r'$e^t$', linewidth=width + 2, color='magenta')
        axs[3].plot(norm(int_errors[n - (n % c.n_steps): n + 1, 1], axis=1),
                    label=r'$e^h$', linewidth=width + 2, color='darkmagenta')

        # Draw trajectories
        axs[0].scatter(*est_target_pos[n - (n % c.n_steps): n + 1].T,
                       color='darkred', linewidth=width + 2, zorder=2)
        axs[0].scatter(*est_arm_pos[n - (n % c.n_steps): n + 1, -1].T,
                       color='darkgreen', linewidth=width + 2, zorder=2)
        axs[0].scatter(*arm_pos[n - (n % c.n_steps): n + 1, -1].T,
                       color='darkblue', linewidth=width + 2, zorder=2)

        # Draw lines
        reached_h = np.where(error_h[n - (n % c.n_steps): n + 1] < c.reach_dist)
        reached_t = np.where(error_t[n - (n % c.n_steps): n + 1] < c.reach_dist)

        for w in range(1, 4):
            axs[w].plot(np.full(c.n_steps, 0), linewidth=width, color='grey')
            axs[w].legend(loc='upper right')

            if reached_h[0].size > 0:
                axs[w].axvline(reached_h[0][0], linestyle='--',
                               linewidth=width + 2,  color='darkblue')
            if reached_t[0].size > 0:
                axs[w].axvline(reached_t[0][0], linestyle='--',
                               linewidth=width + 2,  color='red')

            phase = np.nonzero(mode[n - (n % c.n_steps): n + 1])
            if phase[0].size > 0:
                axs[w].axvline(phase[0][0], linewidth=width)

    start = time.time()
    ani = animation.FuncAnimation(fig, animate, n_t, interval=40)
    writer = animation.writers['ffmpeg'](fps=40)
    ani.save('plots/video.mp4', writer=writer)
    print('\nTime elapsed:', time.time() - start)
    # animate(c.n_steps - 1)
    # plt.savefig('plots/frame')
