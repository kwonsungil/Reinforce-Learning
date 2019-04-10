import os

GAMMA = 0.9
START_EPSILON = 0.2
END_EPSILON = 0.01
REPLAY_SIZE = 50000
BATCH_SIZE = 16
HIDDEN_SIZE = 20
MAX_EPISODES = 5000
MAX_STEPS = 300
TARGET_UPDATE_INTERVAL = 20

out_dir = os.path.join('./', "runs", 'dqn_2015')
summary_dir = os.path.join(out_dir, "summaries")
checkpoint_dir = os.path.join(out_dir, "checkpoints")
os.makedirs(out_dir, exist_ok=True)
os.makedirs(summary_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)


initial_learning_rate = 0.0001
decay_steps = 1000
decay_rate = 0.95
staircase = True