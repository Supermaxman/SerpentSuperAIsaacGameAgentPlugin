import random
import os

from serpent.machine_learning.reinforcement_learning.agent import Agent

import tensorflow as tf
import numpy as np

from . import agent_storage

class SuperAIsaacAgent(Agent):
    def __init__(self, name, game_inputs, callbacks, seed, train=True):
        super().__init__(name, game_inputs, callbacks, seed)
        self.seed = seed
        self.train = train
        self.checkpoint_dir = os.path.join('D:/Models/AIsaac/models', self.name)
        self.log_dir = os.path.join('D:/Logs/AIsaac/logs', self.name)
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.name)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self.nrof_movement_actions = len(self.game_inputs[0]["inputs"])
        self.nrof_attack_actions = len(self.game_inputs[1]["inputs"])
            
        self.game_frame_height = 68
        self.game_frame_width = 120
        self.game_frame_count = 4

        self.learning_rate = 5e-5
        self.momentum = 0.5
        self.gamma = 0.75
        self.move_entropy_scale = 5e-3
        self.attack_entropy_scale = 5e-3
        self.surrogate_objective_clip = 0.2
        self.value_loss_coefficient = 0.5
        self.batch_size = 64
        self.memory_capacity = 4096
        self.epochs = 2

        # hard-coded whenever I stop and start the program to keep count.
        self.current_episode = 22566
        self.total_wins = 196
        self.save_steps = 128
        self.batch_episode_count = 0
        self.current_reward = 0.0
        self.boss_hp_total = 0.0
        self.isaac_hp_total = 0.0
        self.current_observation = None
        self.observations = []
        self.move_storage = agent_storage.ActionRolloutStorage(self.memory_capacity)
        self.attack_storage = agent_storage.ActionRolloutStorage(self.memory_capacity)
    
    def generate_actions(self, observation, **kwargs):
        frames = [2* (game_frame.frame - 0.5) for game_frame in observation.frames]
        frames = np.stack(frames, axis=2)
        #self.
        # if len(self.observations) > 6:
        #     last_game_frame = observation.frames[-1]
        #     import matplotlib.pyplot as plt
        #     plt.imshow((self.last_game_frame.frame * 255).astype(np.uint8), cmap='gray')
        #     plt.savefig('D:/Google Drive/Code/Python/AIsaac/isaac.png')
        #     exit()
        self.current_observation = frames
        # TODO seperate movement/damage actions
        run_results = self.sess.run(
            [self.move_probs, self.attack_probs, self.move_log_probs, self.attack_log_probs, self.move_values, self.attack_values], 
            feed_dict={self.input_observations : np.expand_dims(self.current_observation, axis=0)})
        current_move_probs, current_attack_probs, current_move_log_probs, current_attack_log_probs, current_move_value, current_attack_value = run_results
        self.current_move_probs, self.current_attack_probs = current_move_probs[0], current_attack_probs[0]
        self.current_move_log_probs, self.current_attack_log_probs = current_move_log_probs[0], current_attack_log_probs[0]
        self.current_move_value, self.current_attack_value = current_move_value[0], current_attack_value[0]
        if self.train:
            self.current_move_action = np.random.choice(self.nrof_movement_actions, 1, p=self.current_move_probs)[0]
            self.current_attack_action = np.random.choice(self.nrof_attack_actions, 1, p=self.current_attack_probs)[0]
        else:
            self.current_move_action = np.argmax(self.current_move_probs)
            self.current_attack_action = np.argmax(self.current_attack_probs)
        
        move_label = self.game_inputs_mappings[0][self.current_move_action]
        move_action = self.game_inputs[0]["inputs"][move_label]
        attack_label = self.game_inputs_mappings[1][self.current_attack_action]
        attack_action = self.game_inputs[1]["inputs"][attack_label]
        actions = []
        #TODO remove this after SerpentAI fixes bug with empty actions
        if len(move_action) > 0:
            actions.append((move_label, move_action, None))
        if len(attack_action) > 0:
            actions.append((attack_label, attack_action, None))
        return actions
    
    def observe(self, move_reward=0.0, attack_reward=0.0, terminal=False, boss_hp=None, isaac_hp=None, **kwargs):
        if self.current_observation is None:
            return None
        if self.callbacks.get("before_observe") is not None:
            self.callbacks["before_observe"]()
        
        move_action_prob = self.current_move_probs[self.current_move_action]
        attack_action_prob = self.current_attack_probs[self.current_attack_action]
        move_action_log_prob = self.current_move_log_probs[self.current_move_action]
        attack_action_log_prob = self.current_attack_log_probs[self.current_attack_action]
        move_label = self.game_inputs_mappings[0][self.current_move_action]
        attack_label = self.game_inputs_mappings[1][self.current_attack_action]
        print(f'Move  : {move_reward:+4.2f}|[{move_action_prob:1.4f}]{move_label}')
        print(f'Attack: {attack_reward:+4.2f}|[{attack_action_prob:1.4f}]{attack_label}')
        
        self.observations.append(self.current_observation)
        self.move_storage.insert(
            self.current_move_action,
            move_action_log_prob,
            self.current_move_value,
            move_reward,
            terminal
        )
        self.attack_storage.insert(
            self.current_attack_action,
            attack_action_log_prob,
            self.current_attack_value,
            attack_reward,
            terminal
        )

        self.current_reward = move_reward + attack_reward
        self.cumulative_reward += self.current_reward

        self.analytics_client.track(event_key="REWARD", data={"reward": self.current_reward, "total_reward": self.cumulative_reward})

        if terminal:
            self.current_episode += 1
            self.batch_episode_count += 1
            self.boss_hp_total += boss_hp
            self.isaac_hp_total += isaac_hp
            if isaac_hp > 0.0 and boss_hp < 1e-8:
                self.total_wins += 1
            observations_count = len(self.observations)
            print(f'Episode: {self.current_episode}|{self.batch_episode_count}|{observations_count}/{self.memory_capacity}')
            if observations_count >= self.memory_capacity:
                print(f'Learn: observations_count={observations_count}')
                observations = np.array(self.observations)
                
                move_rewards, _, move_returns, move_log_probs, move_actions, move_advantages = self.move_storage.batch(self.gamma)
                attack_rewards, _, attack_returns, attack_log_probs, attack_actions, attack_advantages = self.attack_storage.batch(self.gamma)

                batch_indices = np.indices([observations_count])[0]
                batch_loss = 0.0
                batch_count = 0
                for _ in range(self.epochs):
                    np.random.shuffle(batch_indices)
                    # drop off excess observations so all mini-batches are the same size
                    epoch_batch_indices = batch_indices[:self.memory_capacity]
                    nrof_mini_batches = self.memory_capacity // self.batch_size

                    epoch_batch_indices = np.reshape(epoch_batch_indices, [nrof_mini_batches, self.batch_size])
                    for m_idx in range(nrof_mini_batches):
                        minibatch_indices = epoch_batch_indices[m_idx]
                        b_observations = observations[minibatch_indices]
                        b_move_actions = move_actions[minibatch_indices]
                        b_move_returns = move_returns[minibatch_indices]
                        b_move_log_probs = move_log_probs[minibatch_indices]
                        b_move_advantages = move_advantages[minibatch_indices]
                        b_attack_actions = attack_actions[minibatch_indices]
                        b_attack_returns = attack_returns[minibatch_indices]
                        b_attack_log_probs = attack_log_probs[minibatch_indices]
                        b_attack_advantages = attack_advantages[minibatch_indices]

                        _, loss, step, summary = self.sess.run(
                            [self.train_op, self.loss, self.global_step, self.summaries], 
                            feed_dict={
                                self.input_observations : b_observations,
                                self.input_move_actions : b_move_actions,
                                self.input_move_returns : b_move_returns,
                                self.input_move_log_probs : b_move_log_probs,
                                self.input_move_advantages : b_move_advantages,
                                self.input_attack_actions : b_attack_actions,
                                self.input_attack_returns : b_attack_returns,
                                self.input_attack_log_probs : b_attack_log_probs,
                                self.input_attack_advantages : b_attack_advantages,
                                self.is_training : True})
                        self.summary_writer.add_summary(summary, global_step=step)
                        batch_loss += loss
                        batch_count += 1
                        if step % self.save_steps == 0:
                            print('Saving...')
                            self.saver.save(self.sess, self.checkpoint_path, global_step=step, write_meta_graph=False)

                batch_loss /= batch_count
                total_move_reward = np.sum(move_rewards)
                total_attack_reward = np.sum(attack_rewards)
                total_reward = total_move_reward + total_attack_reward
                average_move_episode_reward = total_move_reward / self.batch_episode_count
                average_attack_episode_reward = total_attack_reward / self.batch_episode_count
                average_episode_reward = total_reward / self.batch_episode_count
                average_episode_length = observations_count / self.batch_episode_count
                average_boss_hp = self.boss_hp_total / self.batch_episode_count
                average_isaac_hp = self.isaac_hp_total / self.batch_episode_count
                summaries = []
                summaries.append(tf.Summary.Value(tag='average_move_reward', simple_value=average_move_episode_reward))
                summaries.append(tf.Summary.Value(tag='average_attack_reward', simple_value=average_attack_episode_reward))
                summaries.append(tf.Summary.Value(tag='average_episode_reward', simple_value=average_episode_reward))
                summaries.append(tf.Summary.Value(tag='average_episode_length', simple_value=average_episode_length))
                summaries.append(tf.Summary.Value(tag='average_boss_hp', simple_value=average_boss_hp))
                summaries.append(tf.Summary.Value(tag='average_isaac_hp', simple_value=average_isaac_hp))
                summaries.append(tf.Summary.Value(tag='batch_loss', simple_value=batch_loss))
                summaries.append(tf.Summary.Value(tag='total_wins', simple_value=self.total_wins))
                summary = tf.Summary(value=summaries)
                self.summary_writer.add_summary(summary, global_step=step)
                print(f'Step {step}: batch_loss={batch_loss:.4f}')
                self.step = step

                self.analytics_client.track(event_key="TOTAL_REWARD", data={"reward": self.cumulative_reward})
                self.observations = []
                self.move_storage.reset()
                self.attack_storage.reset()
                self.batch_episode_count = 0
                self.boss_hp_total = 0.0
                self.isaac_hp_total = 0.0

        self.current_observation = None
        if self.callbacks.get("after_observe") is not None:
            self.callbacks["after_observe"]()

    def build(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        self.global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), 
            dtype=tf.int32, trainable=False)
        self.is_training = tf.placeholder_with_default(False, [])
        self.input_observations = tf.placeholder(tf.float32, [None, self.game_frame_height, self.game_frame_width, self.game_frame_count])
        self.input_move_returns = tf.placeholder(tf.float32, [None])
        self.input_move_actions = tf.placeholder(tf.int32, [None])
        self.input_move_log_probs = tf.placeholder(tf.float32, [None])
        self.input_move_advantages = tf.placeholder(tf.float32, [None])
        self.input_attack_returns = tf.placeholder(tf.float32, [None])
        self.input_attack_actions = tf.placeholder(tf.int32, [None])
        self.input_attack_log_probs = tf.placeholder(tf.float32, [None])
        self.input_attack_advantages = tf.placeholder(tf.float32, [None])

        self.activation_fn = tf.nn.relu
        self.kernel_initializer = tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='normal')
        with tf.variable_scope('network'):
            net = self.input_observations
            print(net.get_shape())
            net = tf.layers.conv2d(net, filters=64, kernel_size=8, strides=4, padding='valid', kernel_initializer=self.kernel_initializer)
            net = self.activation_fn(net)
            print(net.get_shape())
            net = tf.layers.conv2d(net, filters=128, kernel_size=4, strides=2, padding='same', kernel_initializer=self.kernel_initializer)
            net = self.activation_fn(net)
            print(net.get_shape())
            net = tf.layers.conv2d(net, filters=256, kernel_size=4, strides=2, padding='same', kernel_initializer=self.kernel_initializer)
            net = self.activation_fn(net)
            print(net.get_shape())
            net = tf.layers.conv2d(net, filters=512, kernel_size=4, strides=2, padding='same', kernel_initializer=self.kernel_initializer)
            net = self.activation_fn(net)
            print(net.get_shape())
            net = tf.layers.conv2d(net, filters=1024, kernel_size=3, strides=1, padding='same', kernel_initializer=self.kernel_initializer)
            net = self.activation_fn(net)
            print(net.get_shape())
            cnn_out = tf.layers.flatten(net)
            print(net.get_shape())
            
            prob_net = tf.layers.dense(cnn_out, units=self.nrof_movement_actions + self.nrof_attack_actions)
            print(net.get_shape())
            self.move_logits = prob_net[:, :self.nrof_movement_actions]
            print(self.move_logits.get_shape())
            self.attack_logits = prob_net[:, self.nrof_movement_actions:]
            print(self.attack_logits.get_shape())
            value_net = tf.layers.dense(cnn_out, units=2)
            self.move_values = value_net[:, 0]
            self.attack_values = value_net[:, 1]
            self.move_probs = tf.nn.softmax(self.move_logits, axis=-1)
            self.attack_probs = tf.nn.softmax(self.attack_logits, axis=-1)
            self.move_action = tf.argmax(self.move_logits, axis=-1)
            self.attack_action = tf.argmax(self.attack_logits, axis=-1)
            self.move_log_probs = tf.log(self.move_probs + 1e-8)
            self.attack_log_probs = tf.log(self.attack_probs + 1e-8)
        
        def policy_loss(input_actions, nrof_actions, log_probs, input_log_probs, input_advantages, surrogate_obj_clip):
            next_log_probs = log_probs * tf.one_hot(indices=input_actions, depth=nrof_actions)
            next_log_probs = tf.reduce_sum(next_log_probs, axis=-1)
            ratio = tf.exp(next_log_probs - input_log_probs)
            surr1 = ratio * input_advantages
            surr2 = tf.clip_by_value(ratio, 1.0 - surrogate_obj_clip, 1.0 + surrogate_obj_clip) * input_advantages
            policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            policy_clipped_ratio = tf.reduce_mean(tf.cast(surr2 < surr1, tf.float32))
            return policy_loss, policy_clipped_ratio

        def value_loss(values, input_returns, scale):
            value_loss = scale * tf.losses.mean_squared_error(
                predictions=values,
                labels=input_returns
            )
            return value_loss

        def entropy_loss(probs, scale):
            entropy_loss = scale * tf.reduce_mean(tf.reduce_sum((probs * tf.log(probs + 1e-8)), axis=-1))
            return entropy_loss

        self.move_policy_loss, self.move_policy_clipped_ratio = policy_loss(
            self.input_move_actions,
            self.nrof_movement_actions,
            self.move_log_probs,
            self.input_move_log_probs,
            self.input_move_advantages,
            self.surrogate_objective_clip)
        self.move_value_loss = value_loss(self.move_values, self.input_move_returns, self.value_loss_coefficient)
        self.move_entropy_loss = entropy_loss(self.move_probs, self.move_entropy_scale)
        self.move_loss = self.move_policy_loss + self.move_value_loss
        
        self.attack_policy_loss, self.attack_policy_clipped_ratio = policy_loss(
            self.input_attack_actions,
            self.nrof_attack_actions,
            self.attack_log_probs,
            self.input_attack_log_probs,
            self.input_attack_advantages,
            self.surrogate_objective_clip)
        self.attack_value_loss = value_loss(self.attack_values, self.input_attack_returns, self.value_loss_coefficient)
        self.attack_entropy_loss = entropy_loss(self.attack_probs, self.attack_entropy_scale)
        self.attack_loss = self.attack_policy_loss + self.attack_value_loss

        self.action_loss = self.move_loss + self.attack_loss
        self.entropy_loss = self.move_entropy_loss + self.attack_entropy_loss
        self.total_policy_clipped_ratio = (self.move_policy_clipped_ratio + self.attack_policy_clipped_ratio) / 2.0
        self.loss = self.action_loss + self.entropy_loss
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.momentum)
            gradients_and_vars = self.optimizer.compute_gradients(self.loss)
            gradients, variables = zip(*gradients_and_vars)
            self.max_grad = tf.reduce_max([tf.reduce_max(grad) for grad in gradients])
            self.train_op = self.optimizer.apply_gradients(zip(gradients, variables), self.global_step)
        summaries = []
        summaries.append(tf.summary.scalar('loss', self.loss))
        summaries.append(tf.summary.scalar('action_loss', self.action_loss))
        summaries.append(tf.summary.scalar('move_loss', self.move_loss))
        summaries.append(tf.summary.scalar('attack_loss', self.attack_loss))
        summaries.append(tf.summary.scalar('move_policy_loss', self.move_policy_loss))
        summaries.append(tf.summary.scalar('attack_policy_loss', self.attack_policy_loss))
        summaries.append(tf.summary.scalar('move_value_loss', self.move_value_loss))
        summaries.append(tf.summary.scalar('attack_value_loss', self.attack_value_loss))
        summaries.append(tf.summary.scalar('move_entropy_loss', self.move_entropy_loss))
        summaries.append(tf.summary.scalar('attack_entropy_loss', self.attack_entropy_loss))
        summaries.append(tf.summary.scalar('move_policy_clipped_ratio', self.move_policy_clipped_ratio))
        summaries.append(tf.summary.scalar('attack_policy_clipped_ratio', self.attack_policy_clipped_ratio))
        summaries.append(tf.summary.scalar('total_policy_clipped_ratio', self.total_policy_clipped_ratio))
        summaries.append(tf.summary.scalar('entropy_loss', self.entropy_loss))
        summaries.append(tf.summary.scalar('max_grad', self.max_grad))
        # summaries.append(tf.summary.scalar('global_grad_norm', self.global_grad_norm))
        self.summaries = tf.summary.merge(summaries)
        self.saver = tf.train.Saver(max_to_keep=3, save_relative_paths=True)

    def load(self):
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        #pylint: disable=E1101
        if ckpt and ckpt.model_checkpoint_path:
            self.start_session()
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise FileNotFoundError('No checkpoint file found!')
        self.step = self.sess.run(self.global_step)


    def start_session(self):
        config = tf.ConfigProto()
        #pylint: disable=E1101
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)
        self.summary_writer = tf.summary.FileWriter(
            self.log_dir, 
            self.sess.graph)

    def init(self):
        self.start_session()
        self.sess.run(tf.global_variables_initializer())
        self.step = self.sess.run(self.global_step)

    def close(self):
        self.sess.close()
        tf.reset_default_graph()

    def count_trainable_variables(self):
        total_variables = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape().as_list()
            variable_parameters = 1
            for dim_val in shape:
                variable_parameters *= dim_val
            print('{} | {} -> {}'.format(variable.name, shape, variable_parameters))
            total_variables += variable_parameters
        print('Total number of trainable variables: {}'.format(total_variables))


