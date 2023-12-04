import numpy as np
import utils
import config as c
from network.vae import VAE


# Define agent class
class Agent:
    def __init__(self, limits):
        self.limits = limits

        # Load network
        self.vae = VAE()
        self.vae.load()

        # Initialize intention matrices
        z, i = np.zeros((c.n_joints, c.n_joints)), np.eye(c.n_joints)
        self.I = [np.block([[z, z, z], [i, i, z], [z, z, i]]),
                  np.block([[z, z, z], [z, i, z], [i, z, i]])]

        # Initialize sensory matrices
        self.G_p = np.eye(c.n_joints * 3, c.n_joints)

        # Initialize belief and action
        self.mu = np.zeros((c.n_orders, c.n_joints * 3))
        self.mu_dot = np.zeros_like(self.mu)

        self.a = np.zeros(c.n_joints)
        self.a_dot = np.zeros_like(self.a)

        self.E_i = np.zeros((2, c.n_joints * 3))

        # Initialize precisions
        self.pi_s = [c.pi_prop, np.concatenate(
            (np.tile(c.pi_vis_arm, c.n_joints),
             np.tile(c.pi_vis_target, c.n_joints)))]

        self.alpha = [1 - c.alpha, np.concatenate((np.tile(
            c.alpha,  c.n_joints), np.tile(1, c.n_joints)))]
        self.beta = [1 - c.beta, c.beta]

        # Initialize parameters for phases
        self.mode = 0
        self.slope = np.ones((5, c.n_joints))

    def get_p(self):
        """
        Get predictions
        """
        input_, output = self.vae.predict_visual(self.mu[0, :c.n_joints * 2])
        P = [self.mu[0].dot(self.G_p), output.detach().cpu().numpy()]

        return P, [input_, output], self.mu[1].dot(self.G_p)

    def get_h(self):
        """
        Get intentions
        """
        return np.array([self.mu[0].dot(i) for i in self.I])

    def get_e_s(self, S, P):
        """
        Get sensory prediction errors
        :param S: observations
        :param P: predictions
        """
        return [s - p for s, p in zip(S, P)]

    def get_e_mu(self, H):
        """
        Get dynamics prediction errors
        :param H: intentions
        """
        self.E_i = (H - self.mu[0]) * c.lmbda * self.mode

        return self.mu[1] - self.E_i

    def get_likelihood(self, E_s, grad_v):
        """
        Get likelihood components
        :param E_s: sensory prediction errors
        :param grad_v: visual gradient
        """
        lkh = {}

        lkh['prop'] = self.alpha[0] * self.pi_s[0] * E_s[0].dot(self.G_p.T)
        lkh['vis'] = self.alpha[1] * self.pi_s[1] * self.vae.get_grad(
            *grad_v, E_s[1])
        lkh['vis'] = np.concatenate((lkh['vis'], np.zeros(c.n_joints)))

        return lkh

    def get_mu_dot(self, lkh, E_mu):
        """
        Get belief update
        :param lkh: likelihood components
        :param E_mu: dynamics prediction errors
        """
        self.mu_dot = np.zeros((c.n_orders, c.n_joints * 3))

        # Intention components
        forward_i = np.zeros((c.n_joints * 3))
        # backward_i = np.zeros((c.n_joints * 3))
        for g, i, e in zip(self.beta, self.I, E_mu):
            forward_i += g * e
            # backward_i -= e.dot(c.lmbda * g * (1 - i.T))

        self.mu_dot[0] = self.mu[1] + lkh['prop'] + lkh['vis']
        # self.mu_dot[0] += backward_i
        self.mu_dot[1] = - forward_i
        # self.mu_dot[1] += (e_vel * c.pi_vel).dot(self.G_p.T)

    def get_a_dot(self, e_p):
        """
        Get action update
        :param e_p: proprioceptive error
        """
        self.a_dot = -c.dt * e_p
        # self.a_dot -= e_vel * c.pi_vel

    def integrate(self):
        """
        Integrate with gradient descent
        """
        # Update belief
        self.mu[0] += c.dt * self.mu_dot[0]
        self.mu[0] = np.clip(self.mu[0], 0, 1)
        self.mu[1] += c.dt * self.mu_dot[1]

        # Update action
        self.a += c.dt * self.a_dot
        self.a = np.clip(self.a, -c.a_max, c.a_max)

    def init_belief(self, arm_joint, target_joint, trial):
        """
        Initialize belief
        :param arm_joint: initial arm joint angles
        :param target_joint: initial target joint angles
        :param trial: current trial
        """
        if trial == 0:
            joint_norm = utils.normalize(arm_joint, self.limits)
            target_norm = utils.normalize(target_joint, self.limits)
            hb_norm = utils.normalize(c.home, self.limits)

            self.mu[0] = np.concatenate((joint_norm, target_norm, hb_norm))

        if c.task == 'all':
            self.beta = [1 - c.beta, c.beta] if \
                trial % 2 == 0 else [c.beta, 1 - c.beta]

    def switch_mode(self, step):
        """
        Switch to active inference
        :param step: current step
        """
        if step == 0:
            self.mode = 0
            self.slope = np.ones((30, c.n_joints))
        if not self.mode:
            self.a = np.zeros(c.n_joints)
            self.a_dot = np.zeros(c.n_joints)

        if c.phases == 'fixed' and step == int(c.n_steps / c.phases_ratio):
            self.mode = 1
        elif c.phases == 'dynamic' and not self.mode:
            self.slope = np.concatenate(
                (self.slope[1:], [self.mu_dot[0, c.n_joints: c.n_joints * 2]]))
            if (np.linalg.norm(self.slope, axis=1) < 0.008).all() or \
                    step == int(c.n_steps / c.phases_ratio):
                self.mode = 1
        elif c.phases == 'immediate':
            self.mode = 1

    def inference_step(self, S, target_joint, step):
        """
        Run an inference step
        :param S: observations
        :param target_joint: target joint angles
        :param step: current step
        """
        # Get predictions
        P, grad_v, p_vel = self.get_p()

        # Get intentions
        H = self.get_h()

        # Get sensory prediction errors
        E_s = self.get_e_s(S, P)
        # e_vel = S[1][0] - p_vel

        # Get dynamics prediction errors
        E_mu = self.get_e_mu(H)

        # Get likelihood components
        likelihood = self.get_likelihood(E_s, grad_v)

        # Get belief update
        self.get_mu_dot(likelihood, E_mu)

        # Get action update
        # self.get_a_dot(likelihood.dot(self.G_p))
        self.get_a_dot(E_s[0] * self.pi_s[0] * self.alpha[0])

        # Update
        self.integrate()

        # Start action
        self.switch_mode(step)

        return utils.denormalize(self.a, self.limits) - self.limits[0]
