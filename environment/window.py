import numpy as np
import pyglet
from pyglet.shapes import Circle, Rectangle
from PIL import Image
import utils
import config as c
from environment.arm import Arm


# Define window class
class Window(pyglet.window.Window):
    def __init__(self):
        super().__init__(c.width, c.height, 'Flexible intentions', vsync=False)
        # Initialize arm
        self.arm = Arm()

        # Initialize target
        self.target_joint = np.zeros(c.n_joints)
        self.target_pos = np.zeros(2)
        self.target_dir = np.zeros(2)

        # Initialize agent
        self.agent = None

        # Initialize simulation variables
        self.step, self.trial, self.success = 0, 0, 0

        self.keys = set()
        self.batch = pyglet.graphics.Batch()
        self.offset = (c.width / 2 + c.off_x, c.off_y)

        # Set background
        pyglet.gl.glClearColor(1, 1, 1, 1)

    def on_key_press(self, sym, mod):
        self.keys.add(sym)

    def on_key_release(self, sym, mod):
        self.keys.remove(sym)

    def on_draw(self):
        self.clear()
        objects = self.draw_screen()
        self.batch.draw()

    # Update function to override
    def update(self, dt):
        pass

    # Run simulation with custom update function
    def run(self):
        if c.fps == 0:
            pyglet.clock.schedule(self.update)
        else:
            pyglet.clock.schedule_interval(self.update, 1 / c.fps)
        pyglet.app.run()

    # Stop simulation
    def stop(self):
        pyglet.app.exit()
        self.close()

    # Draw screen
    def draw_screen(self):
        objects = set()

        # Move coordinates on screen
        target_w = self.target_pos + self.offset
        pos_w = self.arm.poses[:, :2] + self.offset

        # Draw target
        # for target in c.targets:
        #     target_pos = self.arm.kinematics(target)[-1, :2]
        #     target_w = np.array(target_pos) + self.offset
        #     objects.add(Circle(*target_w, c.target_size, segments=20,
        #                        color=(255, 0, 0), batch=self.batch))
        if c.task != 'all' or c.task == 'all' and self.trial % 2 != 0:
            objects.add(Circle(*target_w, c.target_size, segments=20,
                               color=(255, 0, 0), batch=self.batch))

        # Draw arm
        objects.add(Circle(*self.offset, 10, segments=20,
                           color=(0, 0, 255), batch=self.batch))

        for j in range(c.n_joints):
            objects = self.draw_arm(objects, j, pos_w[j + 1])

        return objects

    # Draw arm
    def draw_arm(self, objects, n, pos):
        length, width = self.arm.size[n]

        # Draw link
        link = Rectangle(*pos, length, width,
                         color=(0, 0, 255), batch=self.batch)
        link.anchor_position = (length, width / 2)

        link.rotation = -self.arm.poses[n + 1, 2]
        objects.add(link)

        # Draw joint
        objects.add(Circle(*pos, width / 2, segments=20,
                           color=(0, 0, 255), batch=self.batch))

        return objects

    # Get visual observation
    def get_visual_obs(self):
        # Read pixels from screen
        buffer = (pyglet.gl.GLubyte * (3 * c.width * c.height))(0)
        pyglet.gl.glReadPixels(0, 0, c.width, c.height, pyglet.gl.GL_RGB,
                               pyglet.gl.GL_UNSIGNED_BYTE, buffer)

        # Convert to image
        image = Image.frombytes(mode='RGB', size=(c.width, c.height),
                                data=buffer)
        image = np.array(image.transpose(Image.FLIP_TOP_BOTTOM))

        # Normalize and convert white pixels
        image[np.where((image == (255, 255, 255)).all(axis=2))] = (0, 0, 0)

        return image.reshape((3, c.height, c.width)) / 255.0

    # Get proprioceptive observation
    def get_joint_obs(self):
        angles_noise = utils.add_gaussian_noise(self.arm.angles, c.w_p)
        return utils.normalize(angles_noise, self.arm.limits)

    # Get velocity observation
    def get_vel_obs(self):
        vel_noise = utils.add_gaussian_noise(
            self.arm.vel + self.arm.limits[0], c.w_vel)
        return utils.normalize(vel_noise, self.arm.limits)

    # Check if task is successful
    def task_done(self):
        return np.linalg.norm(self.target_pos -
                              self.arm.poses[-1, :2]) < c.reach_dist

    # Generate target randomly or from list
    def sample_target(self, trajectory=None):
        # Sample position
        if trajectory:
            self.target_joint = trajectory[self.trial]
        else:
            self.target_joint = np.random.uniform(*self.arm.limits)

        self.target_pos = self.arm.kinematics(self.target_joint)[-1, :2]

        # Sample velocity
        angle = np.random.rand() * 2 * np.pi
        self.target_dir = np.array((np.cos(angle), np.sin(angle)))

    # Move target
    def move_target(self):
        self.target_pos += c.target_vel * self.target_dir

        # Bounce
        target_w = self.target_pos + self.offset
        if not c.target_size < target_w[0] < c.width - c.target_size:
            self.target_dir = -self.target_dir
        if not c.target_size < target_w[1] < c.height - c.target_size:
            self.target_dir = -self.target_dir

