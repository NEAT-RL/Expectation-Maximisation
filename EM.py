from __future__ import print_function

from multiprocessing import Pool
import os
import argparse
import logging
import random
import sys
import uuid
import gym
import gym.wrappers as wrappers
import configparser
import visualize
from NEATAgent.NEATEMAgent import NeatEMAgent
import numpy as np
from datetime import datetime
import csv
import multiprocessing
import heapq
from time import sleep


class EM(object):

    def __init__(self):
        self.trajectories = []
        self.num_actions = props.getint('policy', 'num_actions')
        self.max_steps = props.getint('train', 'max_steps')
        self.step_size = props.getint('train', 'step_size')
        self.num_trajectories = props.getint('trajectory', 'trajectory_size')
        self.best_trajectory_reward = props.getint('trajectory', 'best_trajectory_reward')
        self.experience_replay = props.getint('evaluation', 'experience_replay')
        self.__initialise_trajectories()
        self.__initialise_agents()

    def __initialise_agents(self):
        self.agent = NeatEMAgent(props.getint('feature', 'dimension'),
                                 props.getint('policy', 'num_actions'),
                                 props.getint('state', 'state_length'),
                                 props.getint('state', 'feature_size'))

    def __initialise_trajectories(self):
        """
        Initialise trajectories of size: 'size'.
        Each trajectory we store N number of state transitions. (state, reward, next state, action)
        :param:
        :return:
        """
        logger.debug("Creating trajectories for first time...")
        t_start = datetime.now()
        num_actions = self.num_actions
        for x in range(self.num_trajectories):
            self.trajectories.append(EM.initialise_trajectory(num_actions))
        # self.trajectories = heapq.nlargest(len(self.trajectories), self.trajectories)
        self.trajectories.sort(reverse=True)  # We want to do in-place sorting for memory saving. sorting(n) is faster than heapq.nlargest for larger values of n.
        logger.debug("Finished: Creating trajectories. Time taken: %f", (datetime.now() - t_start).total_seconds())

    @staticmethod
    def initialise_trajectory(num_actions):
        max_steps = props.getint('train', 'max_steps')
        step_size = props.getint('train', 'step_size')
        state_starts = []
        state_ends = []
        rewards = []
        actions = []
        state = env.reset()
        terminal_reached = False
        steps = 0
        total_reward = 0
        while not terminal_reached and steps < max_steps:
            # sample action from the environment
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)

            for x in range(step_size - 1):
                if done:
                    terminal_reached = True
                    break
                next_state, reward2, done, info = env.step(action)
                reward += reward2
            action_array = np.zeros((num_actions,))
            action_array[action] = 1
            state_starts.append(state)
            state_ends.append(next_state)
            rewards.append(reward)
            actions.append(action_array)
            # insert state transition to the trajectory
            total_reward += reward
            state = next_state
            steps += 1
            if done:
                terminal_reached = True
        return total_reward, uuid.uuid4(), state_starts, state_ends, actions, rewards

    def execute_algorithm(self):
        iterations = props.getint('train', 'iterations')
        for i in range(iterations):
            logger.debug("Running iteration %d", i)
            t_start = datetime.now()
            self.agent.fitness = self.fitness_function()
            logger.debug("Agent fitness: %f", self.agent.fitness)
            if i % 10 == 0:
                sleep(5)
            if i % 50 == 0:
                print(self.agent.get_policy().parameters)
                test_agent(self.agent, i)
            logger.debug("Completed Iteration %d. Time taken: %f", i, (datetime.now() - t_start).total_seconds())

    def fitness_function(self):
        """
        Generate trajectory.
        Insert into Trajectories.
        Select best trajectory and perform policy update
        :return fitness of agent:
        """
        # worst_trajectories = self.trajectories[int(0.9 * self.num_trajectories):]
        # self.trajectories = self.trajectories[0: int(0.9 * self.num_trajectories)] + [random.choice(worst_trajectories) for x in range(int(0.1 * self.num_trajectories))]
        # self.trajectories.sort(reverse=True)
        self.trajectories = heapq.nlargest(self.num_trajectories, self.trajectories)
        logger.debug("Worst Trajectory reward: %f", self.trajectories[len(self.trajectories) - 1][0])
        logger.debug("Best Trajectory reward: %f", self.trajectories[0][0])

        # if self.trajectories[0][0] >= self.best_trajectory_reward:
        #     # Found the best possible trajectory so now turn policies into greedy one
        #     for agent in agents:
            # self.agent.policy.is_greedy = True
        #     greedy = True

        all_state_starts = []
        all_state_ends = []
        all_rewards = []
        all_actions = []

        for i, (_, _, state_starts, state_ends, actions, rewards) in enumerate(self.trajectories):
            all_actions += actions
            all_rewards += rewards
            all_state_starts += state_starts
            all_state_ends += state_ends

        len_state_transitions = len(all_state_starts)
        random_indexes = random.sample(range(0, len_state_transitions), self.experience_replay if len_state_transitions > self.experience_replay else len_state_transitions)
        # random_state_transitions = random.sample(state_transitions, self.experience_replay) if len(state_transitions) > self.experience_replay else state_transitions

        # update value function
        logger.debug("Updating value function")
        self.agent.update_value_function(random_indexes, all_state_starts, all_state_ends, all_rewards)

        # update policy parameter
        logger.debug("Updating policy function")

        self.agent.update_policy_function_theano(all_state_starts, all_state_ends, all_actions, all_rewards)
        # self.agent.update_policy_function(random_state_transitions, state_transitions, self.pool)

        new_trajectories = []
        if allow_multiprocessing:
            results = [
                pool.apply_async(self.generate_new_trajectory) for
                i
                in
                range(2)]
            new_trajectories = [new_trajectory.get() for new_trajectory in results]

        else:
            for i in range(2):
                new_trajectories.append(self.generate_new_trajectory())

        # Calculate the average of new trajectories and if its better then the best average then save policy parameters of agent
        average_reward = 0
        for i in range(len(new_trajectories)):
            average_reward += new_trajectories[i][0]
        #
        average_reward /= len(new_trajectories)
        if average_reward >= self.best_trajectory_reward:
            logger.debug('Will perform kl divergence checks now')
            self.agent.get_policy().check_kl_divergence = True

        self.trajectories += new_trajectories
        # order the trajectories
        self.trajectories.sort(reverse=True)

        # now assign fitness to each individual/genome
        # fitness is the log prob of following the best 5 trajectory
        best_trajectories = self.trajectories[:5]
        best_start_states = []
        best_actions = []

        for i, (_, _, state_starts, state_ends, actions, rewards) in enumerate(best_trajectories):
            best_start_states += state_starts
            best_actions += actions

        best_trajectory_prob = self.agent.calculate_agent_fitness(best_start_states, best_actions)

        return best_trajectory_prob

    def generate_new_trajectory(self):
        #logger.debug("Generating new trajectory")
        num_actions = self.num_actions
        state_starts = []
        state_ends = []
        rewards = []
        actions = []

        # perform a rollout
        state = env.reset()
        terminal_reached = False
        steps = 0
        total_reward = 0
        while not terminal_reached and steps < self.max_steps:
            state_features = self.agent.feature.phi(state)
            # get recommended action and the action distribution using policy
            action, actions_distribution = self.agent.get_policy().get_action_theano(state_features)
            next_state, reward, done, info = env.step(action)

            for x in range(self.step_size - 1):
                if done:
                    terminal_reached = True
                    break
                next_state, reward2, done, info = env.step(action)
                reward += reward2

            action_array = np.zeros((num_actions,))
            action_array[action] = 1
            state_starts.append(state)
            state_ends.append(next_state)
            rewards.append(reward)
            actions.append(action_array)

            total_reward += reward
            state = next_state
            steps += 1
            if done:
                terminal_reached = True

        # logger.debug("Finished: Generating new trajectory")
        logger.debug("total_reward: %f", total_reward)
        return total_reward, uuid.uuid4(), state_starts, state_ends, actions, rewards


def test_agent(agent, iteration_count):
    t_start = datetime.now()

    test_episodes = props.getint('test', 'test_episodes')
    max_steps = props.getint('test', 'max_steps')
    step_size = props.getint('test', 'step_size')

    total_steps = 0.0
    total_rewards = 0.0
    # Perform testing in parallel
    # use the agents best policy parameter
    # agent.get_policy().set_policy_parameters(agent.best_policy_parameters)
    for i in range(test_episodes):
        state = env.reset()
        terminal_reached = False
        steps = 0
        while not terminal_reached and steps < max_steps:
            if display_game:
                env.render()
            state_features = agent.feature.phi(state)
            action, actions_distribution = agent.get_policy().get_action_theano(state_features)
            state, reward, done, info = env.step(action)
            total_rewards += reward

            for x in range(step_size - 1):
                if done:
                    terminal_reached = True
                    break
                state, reward, done, info = env.step(action)
                total_rewards += reward

            steps += 1
            if done:
                terminal_reached = True

        total_steps += steps
    average_steps_per_episodes = total_steps / test_episodes
    average_rewards_per_episodes = total_rewards / test_episodes

    # save this to file
    entry = [iteration_count, average_steps_per_episodes, average_rewards_per_episodes]
    with open(r'agent_evaluation-{0}.csv'.format(time), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(entry)

    logger.debug("Finished: evaluating best agent. Time taken: %f", (datetime.now() - t_start).total_seconds())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    parser.add_argument('display', nargs='?', default='false', help='Show display of game. true or false')
    parser.add_argument('--threads', nargs='?', default='max', help='Number of threads to use. 0 means no threads')
    args = parser.parse_args()

    # Call `undo_logger_setup` if you want to undo Gym's logger setup
    # and configure things manually. (The default should be fine most
    # of the time.)
    gym.undo_logger_setup()
    time = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    logging.basicConfig(filename='log/debug-{0}.log'.format(time),
                        level=logging.DEBUG, format='[%(asctime)s] %(message)s')
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # You can set the level to logging.DEBUG or logging.WARN if you
    # want to change the amount of output.
    logger.setLevel(logging.DEBUG)

    env = gym.make(args.env_id)

    logger.debug("action space: %s", env.action_space)
    logger.debug("observation space: %s", env.observation_space)

    # load properties file
    local_dir = os.path.dirname(__file__)
    logger.debug("Loading Properties File")
    props = configparser.ConfigParser()
    prop_path = os.path.join(local_dir, 'properties/{0}/neatem_properties.ini'.format(env.spec.id))
    props.read(prop_path)
    logger.debug("Finished: Loading Properties File")

    processes = None
    allow_multiprocessing = True
    if args.threads == '0':
        allow_multiprocessing = False
    elif not args.threads == 'max':
        processes = int(args.threads)

    print(allow_multiprocessing)
    if allow_multiprocessing:
        pool = multiprocessing.Pool(processes=processes)

    # initialise experiment
    experiment = EM()

    display_game = True if args.display == 'true' else False
    try:
        experiment.execute_algorithm()
    except KeyboardInterrupt:
        logger.debug("User break.")
    finally:
        if allow_multiprocessing:
            pool.terminate()
        env.close()

    # Upload to the scoreboard. We could also do this from another
    # logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    # gym.upload(outdir)
