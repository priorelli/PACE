import seaborn as sns
import numpy as np
import utils
from plots.dynamics import plot_dynamics
from plots.perception import plot_perception
from plots.attractors import plot_attractors
from plots.gradients import plot_gradients
from plots.final_pos import plot_final_pos
from plots.scores import plot_scores
from plots.video import record_video

sns.set_theme(style='darkgrid', font_scale=2)


def main():
    width = 3

    # Parse arguments
    options = utils.get_plot_options()

    # Load log
    log = np.load('simulation/log.npz')

    # Choose plot to display
    if options.dynamics:
        plot_dynamics(log, width)

    elif options.perception:
        plot_perception(log, width)

    elif options.attractors:
        plot_attractors(log, width)

    elif options.gradients:
        plot_gradients()

    elif options.scores:
        plot_scores()

    elif options.video:
        record_video(log, width)

    else:
        plot_final_pos(log)


if __name__ == '__main__':
    main()
