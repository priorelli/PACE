import numpy as np
import utils
import config as c
from environment.window import Window


# Define data generation class
class DataGeneration(Window):
    def __init__(self):
        super().__init__()
        # Initialize dataset
        self.x = np.zeros((c.n_datapoints, c.n_joints * 2))
        self.y = np.zeros((c.n_datapoints, 3, c.height, c.width))

        self.i = 0

    def update(self, dt):
        # Sample random rotation
        arm_joint_norm = np.random.rand(c.n_joints)
        arm_joint = utils.denormalize(arm_joint_norm, self.arm.limits)

        # Sample random target
        self.sample_target()
        c.target_size = np.random.randint(*c.target_min_max)
        target_joint_norm = utils.normalize(self.target_joint, self.arm.limits)

        # Set rotation
        self.arm.set_rotation(arm_joint)

        # Manually update environment
        self.on_draw()

        # Get observation
        visual_obs = self.get_visual_obs()

        # Store sample to dataset
        self.x[self.i] = np.concatenate((arm_joint_norm, target_joint_norm))
        self.y[self.i] = visual_obs
        self.i += 1

        # Save dataset
        if self.i == c.n_datapoints:
            np.savez_compressed('network/dataset', x=self.x, y=self.y)
            self.stop()
