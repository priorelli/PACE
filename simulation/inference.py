import time
import utils
import config as c
from environment.window import Window
from simulation.agent import Agent
from environment.log import Log


# Define inference class
class Inference(Window):
    def __init__(self):
        super().__init__()
        # Initialize agent
        self.agent = Agent(self.arm.limits)

        # Set trajectory for simulation
        self.trajectory = self.set_trajectory()

        # Initialize error tracking
        self.log = Log()

        # Initialize trial
        self.reset_trial()
        self.time = time.time()

    def update(self, dt):
        # Get observations
        S = self.get_joint_obs(), self.get_visual_obs()

        # Perform free energy step
        action = self.agent.inference_step(S, self.target_joint, self.step)

        # Update arm
        action_noise = utils.add_gaussian_noise(action, c.w_a)
        self.arm.update(action_noise)

        # Move objects
        if c.context == 'dynamic':
            self.move_target()

        # Track log
        self.log.track(self.trial - 1, self.step, self.agent,
                       self.arm, self.target_pos)

        # Print info
        self.step += 1
        if self.step % 50 == 0:
            utils.print_info(self.trial, self.success, self.step, dt)

        # Reset trial
        if self.step == c.n_steps:
            self.reset_trial()

    def reset_trial(self):
        self.success += self.task_done()

        if self.trial == c.n_trials:
            # Simulation done
            utils.print_score(self.log, time.time() - self.time)
            self.log.save_log()
            self.stop()
        else:
            if c.task == 'single':
                self.arm.angles = c.home

            # Sample target
            self.sample_target(self.trajectory)

            # Initialize simulation
            self.agent.init_belief(self.arm.angles, self.arm.angles,
                                   self.trial)

            self.step = 0
            self.trial += 1

    def set_trajectory(self):
        if c.task == 'test':
            return None
        elif c.task == 'single':
            trajectory = [c.targets[2] for _ in range(c.n_trials)]
        else:
            trajectory = [(target, c.home) for target in c.targets
                          for _ in range(c.n_reps)]
            trajectory = [item for sublist in trajectory for item in sublist]

        c.n_trials = len(trajectory)
        return trajectory
