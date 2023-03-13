# Window
width = 128
height = 96
off_x = -20
off_y = 22
fps = 0

# Agent
dt = 0.4
a_max = 1.0
alpha = 0.4
beta = 0.0
k = 0.06

w_p = 0  # 2e-3
w_a = 0  # 5e-5
w_vel = 0  # 2e-4

pi_prop = 0.5
pi_vis_arm = 2e-5
pi_vis_target = 3e-4
pi_vel = 0.0

# Network
n_epochs = 100
n_batch = 32
n_datapoints = 20000
learning_rate = 0.001
step_size = 20
gamma = 0.95
variance = 0.02
log_name = ''

# Inference
task = 'test'  # test, all, single
context = 'static'  # static, dynamic
phases = 'fixed'  # immediate, fixed, dynamic

target_size = 5
target_vel = 0.1
target_min_max = (5, 12)
reach_dist = 10

n_trials = 20
n_steps = 500
n_reps = 2
n_orders = 2
phases_ratio = 3
home = (10, 42, 130)
targets = [(8, 119, 0), (10, 95, 0), (0, 46, 65),
           (10, 78, 75), (0, 67, 69), (0, 21, 107),
           (0, 77, 102), (0, 50, 105), (0, 2, 135)]

# Arm
joints = {}
joints['trunk'] = {'link': None, 'angle': home[0],
                   'limit': (0, 10), 'size': (17, 16)}
joints['shoulder'] = {'link': 'trunk', 'angle': home[1],
                      'limit': (-10, 130), 'size': (27, 14)}
joints['elbow'] = {'link': 'shoulder', 'angle': home[2],
                   'limit': (10, 130), 'size': (38, 12)}
n_joints = len(joints)
