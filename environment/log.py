import numpy as np
import utils
import config as c


# Define log class
class Log:
    def __init__(self):
        # Initialize logs
        self.mu = np.zeros((c.n_trials, c.n_steps, c.n_orders, c.n_joints * 3))
        self.mu_dot = np.zeros_like(self.mu)

        self.actions = np.zeros((c.n_trials, c.n_steps, c.n_joints))
        self.actions_dot = np.zeros_like(self.actions)

        self.angles = np.zeros((c.n_trials, c.n_steps, c.n_joints))
        self.est_angles = np.zeros_like(self.angles)

        self.target_pos = np.zeros((c.n_trials, c.n_steps, 2))
        self.est_target_pos = np.zeros_like(self.target_pos)

        self.hand_pos = np.zeros((c.n_trials, c.n_steps, 2))
        self.est_hand_pos = np.zeros_like(self.hand_pos)

        self.mode = np.zeros((c.n_trials, c.n_steps))
        self.int_errors = np.zeros((c.n_trials, c.n_steps, 2, c.n_joints * 3))

    # Track logs for each time step
    def track(self, trial, step, agent, arm, target_pos):
        self.mu[trial, step] = agent.mu
        self.mu_dot[trial, step] = agent.mu_dot

        self.actions[trial, step] = agent.a
        self.actions_dot[trial, step] = agent.a_dot

        limits = np.tile(agent.limits, c.n_joints)
        mu = utils.denormalize(agent.mu[0], limits)
        self.angles[trial, step] = arm.angles
        self.est_angles[trial, step] = mu[:c.n_joints]

        self.target_pos[trial, step] = target_pos
        self.est_target_pos[trial, step] = arm.kinematics(
            mu[c.n_joints: c.n_joints * 2])[-1, :2]

        self.hand_pos[trial, step] = arm.poses[-1, :2]
        self.est_hand_pos[trial, step] = arm.kinematics(
            mu[:c.n_joints])[-1, :2]

        self.mode[trial, step] = agent.mode

        int_errors = agent.E_i * np.repeat(agent.beta, c.n_joints * 3). \
            reshape(2, c.n_joints * 3)
        self.int_errors[trial, step] = int_errors

    # Save log to file
    def save_log(self):
        np.savez_compressed('simulation/log' + c.log_name,
                            mu=self.mu, mu_dot=self.mu_dot,
                            actions=self.actions,
                            actions_dot=self.actions_dot,
                            angles=self.angles,
                            est_angles=self.est_angles,
                            target_pos=self.target_pos,
                            est_target_pos=self.est_target_pos,
                            hand_pos=self.hand_pos,
                            est_hand_pos=self.est_hand_pos,
                            mode=self.mode, int_errors=self.int_errors)
