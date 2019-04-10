import os
import random
from collections import deque

import gym
import numpy as np
import tensorflow as tf

from config import config_dqn_2013 as config


class DQN(object):
    def __init__(self, is_train, game):
        self.is_train = is_train
        self.env = gym.make(game)
        self.replay_buffer = deque()
        # self.time_step = 0
        self.epsilon = -1
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.states = tf.placeholder(tf.float32, shape=[None, self.state_dim], name="states")
            self.acts = tf.placeholder(tf.float32, [None, self.action_dim], name='acts')
            self.rewards = tf.placeholder(tf.float32, [None], name='rewards')
            self.y_target = tf.placeholder(tf.float32, [None])

            self.Q_main = self.create_network('main_2013')

            ConfigProto = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3))
            ConfigProto.gpu_options.allow_growth = True
            self.sess = tf.Session(config=ConfigProto, graph=self.graph)
            self.saver = tf.train.Saver(max_to_keep=40)

            if self.is_train:
                self.epsilon = config.START_EPSILON
                self.cost = self.calculate_loss()
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                # self.learning_rate = tf.train.exponential_decay(
                #     config.initial_learning_rate, self.global_step, config.decay_steps,
                #     config.decay_rate, config.staircase, name='learning_rate')
                self.learning_rate = config.initial_learning_rate
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost,
                                                                                     global_step=self.global_step,
                                                                                     name='optimizer')
                self.summary_writer = tf.summary.FileWriter(config.summary_dir, graph=self.sess.graph)

                # tf.summary.scalar('rewards', tf.reduce_mean(self.rewards))
                tf.summary.scalar('dqn/rewards', tf.reduce_sum(self.rewards))
                tf.summary.scalar("dqn/loss", self.cost)
                tf.summary.scalar("dqn/lr", self.learning_rate)
                self.merged_summary = tf.summary.merge_all()

            filename = tf.train.latest_checkpoint(config.checkpoint_dir)
            self.sess.run(tf.global_variables_initializer())
            if filename is not None:
                print('restore from : ', filename)
                self.saver.restore(self.sess, filename)

    def create_network(self, name):
        with tf.variable_scope(name):
            model = tf.layers.dense(self.states, 64, activation=tf.nn.relu)
            model = tf.layers.dense(model, 32, activation=tf.nn.relu)
            Q_predict = tf.layers.dense(model, self.action_dim, None)
        return Q_predict

    def calculate_loss(self):
        y_predict = tf.reduce_sum(tf.multiply(self.Q_main, self.acts), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(y_predict - self.y_target))
        return cost

    def train(self):
        with self.graph.as_default():
            for episode in range(1, config.MAX_EPISODES):
                state, total_reward, done = self.env.reset(), 0, False
                reward_list = []

                for step in range(1, config.MAX_STEPS):
                    action = self.act(state)
                    next_state, next_reward, done, _ = self.env.step(action)
                    temp_action = [0, 0]
                    temp_action[action] = 1
                    next_reward = -10 if done else 1
                    # total_reward += next_reward
                    self.replay_buffer.append((state, temp_action, next_reward, next_state, done))
                    if done:
                        reward_list.append(step)
                        # print('episode: ', episode, 'step : ', step)
                        break
                    if len(self.replay_buffer) > config.REPLAY_SIZE:
                        self.replay_buffer.popleft()

                    state = next_state

                print('episode: ', episode, 'Evaluation Average Reward:', np.average(np.array([reward_list])))
                # reward_list = []
                if episode % 10 == 0:
                    for _ in range(50):
                        batches = random.sample(self.replay_buffer, config.BATCH_SIZE)
                        states = []
                        actions = []
                        rewards = []
                        next_states = []

                        for data in batches:
                            states.append(data[0])
                            actions.append(data[1])
                            rewards.append(data[2])
                            next_states.append(data[3])
                            # print(states, actions)

                        y_targets = []
                        Q_predicts = self.sess.run(self.Q_main, feed_dict={self.states: next_states})
                        # Q_predicts = self.sess.run(self.Q_main, feed_dict={self.states: next_states})
                        for i in range(0, config.BATCH_SIZE):
                            # print(Q_predicts[i], actions[i])
                            done = batches[i][4]
                            if done:
                                y_targets.append(rewards[i])
                            else:
                                y_targets.append(rewards[i] + config.GAMMA * np.max(Q_predicts[i]))

                        _, global_step, summary, cost = self.sess.run(
                            [self.optimizer, self.global_step, self.merged_summary, self.cost],
                            feed_dict={
                                self.y_target: y_targets,
                                self.acts: actions,
                                self.states: states,
                                # self.rewards: rewards
                                # reward는 10 episode 마다평균
                                self.rewards: reward_list
                            })
                        # print('global_step : ', global_step, cost, y_targets)

                    self.saver.save(self.sess, os.path.join(config.checkpoint_dir, 'model.ckpt'), global_step=global_step)
                    self.summary_writer.add_summary(summary, global_step=global_step)

    def act(self, state):
        # print('self.epsilon : ', self.epsilon)
        if random.random() <= self.epsilon:
            self.epsilon -= (config.START_EPSILON - config.END_EPSILON) / config.MAX_EPISODES
            return random.randint(0, self.action_dim - 1)
        else:
            self.epsilon -= (config.START_EPSILON - config.END_EPSILON) / config.MAX_EPISODES
            return np.argmax(self.sess.run(self.Q_main, feed_dict={self.states: [state]})[0])

    def play(self):
        with self.graph.as_default():
            for episode in range(100):
                state, total_reward, done = self.env.reset(), 0, False
                rewards = 0
                while not done:
                    action = self.act(state)
                    next_state, next_reward, done, _ = self.env.step(action)
                    rewards += 1
                    state = next_state
                print('episode: ', episode, 'Reward:', rewards)


if __name__ == '__main__':
    # mode = 'train'
    mode = 'test'
    game = 'CartPole-v0'
    if mode == 'train':
        agent = DQN(True, game)
        agent.train()
    else:
        agent = DQN(False, game)
        agent.play()