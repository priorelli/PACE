import numpy as np
import matplotlib.pyplot as plt
import utils
import config as c
from network.vae import VAE
from environment.window import Window


def plot_gradients():
    # Create plot
    fig, axs = plt.subplots(2, 2, num='Gradients', figsize=(30, 30))

    # Load network
    network = VAE()
    network.load()

    log = {}

    # Plot gradients
    for i, obj in enumerate(('arm', 'target')):
        for j, coord in enumerate(('cart', 'pol')):
            env = Gradient(network, axs[i, j], obj, coord)
            axs[i, j].set_aspect('equal')
            env.run()
            log[obj + '_' + coord] = env.grads

    # np.savez_compressed('log_grads', arm_cart=log['arm_cart'],
    #                     arm_pol=log['arm_pol'],
    #                     target_cart=log['target_cart'],
    #                     target_pol=log['target_pol'])

    fig.savefig('plots/gradients', bbox_inches='tight')


class Gradient(Window):
    def __init__(self, network, axs, obj, coord):
        super().__init__()
        self.network = network
        self.axs = axs
        self.obj = obj
        self.coord = coord

        self.grads = []

        self.current = np.array([0, 57, 90])
        self.target = np.array([0, 60, 90])

        if self.coord == 'cart':
            self.axs.set_xlabel('x (px)')
            self.axs.set_ylabel('y (px)')
        else:
            self.axs.set_xlabel(r'$\theta_1 (°)$')
            self.axs.set_ylabel(r'$\theta_2 (°)$')

    def update(self, dt):
        # Draw arm and target
        arm_pos = self.arm.kinematics(self.current)[:, :2]
        target_pos = self.arm.kinematics(self.target)[-1, :2]

        if self.coord == 'cart':
            for i in range(c.n_joints):
                self.axs.plot(*arm_pos[i: i + 2].T,
                              linewidth=self.arm.size[i, 1] * 3, color='b')

            self.axs.scatter(*target_pos, color='r', s=2000)
        else:
            color = 'b' if self.obj == 'arm' else 'r'
            self.axs.scatter(*self.target[1:], color=color, s=2000)

        # Draw quiver
        x, u = self.joint(self.current, self.target)
        q = self.axs.quiver(*x.T, *u.T, zorder=-1, angles='xy')

        self.grads = [x, u]
        self.stop()

    def joint(self, arm, target):
        # Get attractor
        self.arm.set_rotation(arm)
        self.on_draw()
        attractor = self.get_visual_obs()

        # Compute gradients
        x, u = [], []
        for j in range(*self.arm.limits[:, 1].astype(int), 10):
            for k in range(*self.arm.limits[:, 2].astype(int), 10):
                # Compute visual prediction
                joint = np.array([0, j, k])
                joint_obs = np.concatenate((joint, target)) if \
                    self.obj == 'arm' else np.concatenate((arm, joint))
                norm = utils.normalize(joint_obs, np.tile(self.arm.limits, 2))
                input_, output = self.network.predict_visual(norm)

                # Get error
                predict = output.detach().cpu().numpy()
                error = attractor.reshape(1, 3, c.height, c.width) - predict

                # Get gradient
                grad = self.network.get_grad(input_, output, error)

                if self.obj == 'arm':
                    grad = grad[:c.n_joints]
                else:
                    grad = grad[c.n_joints:]

                joint_new = np.clip(joint + grad / 20, *self.arm.limits)
                pos = self.arm.kinematics(joint)[-1, :2]
                pos_new = self.arm.kinematics(joint_new)[-1, :2]

                if self.coord == 'cart':
                    x.append(pos)
                    u.append(pos_new - pos)
                else:
                    x.append(joint[1:])
                    u.append(joint_new[1:] - joint[1:])

        return np.array(x), np.array(u)
