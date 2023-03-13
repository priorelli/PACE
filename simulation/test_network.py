import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import utils
import config as c
from network.vae import VAE
from environment.window import Window


# Define data generation class
class Test(Window):
    def __init__(self):
        super().__init__()
        # Load network
        self.vae = VAE()
        self.vae.load()

        self.n_test = int(c.n_datapoints / 4)

        self.dec_acc = np.zeros(self.n_test)
        self.enc_acc = np.zeros(self.n_test)
        self.full_acc = np.zeros(self.n_test)

        self.i = 0

    def update(self, dt):
        # Sample random rotation
        joint_norm = np.random.rand(c.n_joints)
        joint = utils.denormalize(joint_norm, self.arm.limits)

        # Sample random target
        c.target_size = np.random.randint(*c.target_min_max)
        self.sample_target()
        target_norm = utils.normalize(self.target_joint, self.arm.limits)

        # Set rotation
        self.arm.set_rotation(joint)

        # Manually update environment
        self.on_draw()

        # Get observation
        visual_obs = self.get_visual_obs()

        # Predict visual observation
        joint_obs = np.concatenate((joint_norm, target_norm))
        _, output = self.vae.predict_visual(joint_obs)
        y_pred = output.detach().cpu().numpy()

        # Predict joint observation
        output = self.vae.predict_joint(visual_obs)
        x_pred = output.detach().cpu().numpy()

        # Reconstruct observation
        _, output = self.vae.predict_visual(x_pred)
        y_final = output.detach().cpu().numpy()

        # Plot images
        # fig, axs = plt.subplots(1, 3)
        # axs[0].imshow(visual_obs.reshape(c.height, c.width, 3))
        # axs[1].imshow(y_pred.reshape(c.height, c.width, 3))
        # axs[2].imshow(y_final.reshape(c.height, c.width, 3))
        # plt.show()

        # Compute errors
        self.dec_acc[self.i] = norm(y_pred - visual_obs)
        self.enc_acc[self.i] = norm(x_pred - joint_obs)
        self.full_acc[self.i] = norm(y_final - visual_obs)

        self.i += 1

        # Stop simulation
        if self.i == self.n_test:
            print('Decoder accuracy: {:.2f}'.format(np.mean(self.dec_acc)))
            print('Full accuracy: {:.2f}'.format(np.mean(self.full_acc)))
            print('Encoder accuracy: {:.2f}'.format(np.mean(self.enc_acc)))
            self.stop()
