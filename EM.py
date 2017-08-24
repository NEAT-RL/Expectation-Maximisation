from __future__ import print_function

from multiprocessing import Pool
import os
import pickle
import argparse
import logging
import random
import sys
import neat
import gym
import gym.wrappers as wrappers
import configparser
import visualize
from NEATAgent.NEATEMAgent import NeatEMAgent
import numpy as np
import heapq
from datetime import datetime
import csv
class StateTransition(object):
    def __init__(self, start_state, action, reward, end_state):
        self.start_state = start_state
        self.action = action
        self.reward = reward
        self.end_state = end_state

    def __hash__(self):
        return hash(str(np.concatenate((self.start_state, self.end_state))))

    def __eq__(self, other):
        return np.array_equal(self.start_state, other.start_state) and np.array_equal(self.end_state, other.end_state)

    def get_start_state(self):
        return self.start_state

    def get_end_state(self):
        return self.end_state

    def get_action(self):
        return self.action

    def get_reward(self):
        return self.reward

    def get_tuple(self):
        return self.start_state, self.action, self.reward, self.end_state


class EM(object):

    def __init__(self, pool):
        self.pool = pool
        self.trajectories = []
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
        num_trajectories = props.getint('initialisation', 'trajectory_size')

        results = [self.pool.apply_async(EM.initialise_trajectory) for x in range(num_trajectories)]
        results = [trajectory.get() for trajectory in results]
        self.trajectories = results
        logger.debug("Finished: Creating trajectories. Time taken: %f", (datetime.now() - t_start).total_seconds())

    @staticmethod
    def initialise_trajectory():
        max_steps = props.getint('train', 'max_steps')
        step_size = props.getint('train', 'step_size')
        trajectory = []
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

            state_transition = StateTransition(state, action, reward, next_state)
            # insert state transition to the trajectory
            trajectory.append(state_transition)
            total_reward += reward
            state = next_state
            steps += 1
            if done:
                terminal_reached = True

        return total_reward, datetime.now(), trajectory

    def execute_algorithm(self):
        iterations = props.getint('train', 'iterations')
        for i in range(iterations):
            logger.debug("Running iteration %d", i)
            t_start = datetime.now()
            self.agent.fitness = self.fitness_function()
            logger.debug("Agent fitness: %f", self.agent.fitness)
            if i % 200 == 0:
                test_agent(self.agent, i)
            logger.debug("Completed Iteration %d. Time taken: %f", i, (datetime.now() - t_start).total_seconds())

    def fitness_function(self):
        """
        Generate trajectory.
        Insert into Trajectories.
        Select best trajectory and perform policy update
        :return fitness of agent:
        """
        total_reward, new_state_transitions = self.generate_new_trajectory()

        self.trajectories.append((total_reward, datetime.now(), new_state_transitions))

        # strip weak trajectories from trajectory_set
        self.trajectories = heapq.nlargest(props.getint('initialisation', 'trajectory_size'), self.trajectories)

        # Collect set of state transitions
        state_transitions = set()
        for i in range(len(self.trajectories)):
            state_transitions = state_transitions | set(self.trajectories[i][2])

        experience_replay = props.getint('evaluation', 'experience_replay')
        random_state_transitions = random.sample(state_transitions, experience_replay)

        # update value function
        logger.debug("Updating value function")
        self.agent.update_value_function(random_state_transitions)

        # update policy parameter
        logger.debug("Updating policy function")
        self.agent.update_policy_func tion(self.trajectories, state_transitions, self.pool)

        # now assign fitness to each individual/genome
        # fitness is the log prob of following the best trajectory
        best_trajectory = self.trajectories[0]
        best_trajectory_prob = 0

        total_reward, _, trajectory_state_transitions = best_trajectory
        for j, state_transition in enumerate(trajectory_state_transitions):
            # calculate probability of the action where policy action = action
            state_features = self.agent.feature.phi(state_transition.get_start_state())
            _, actions_distribution = self.agent.get_policy().get_action(state_features)
            if actions_distribution[state_transition.get_action()] <= 0:
                logger.error("Negative Probabilities!!!")
                logger.error(actions_distribution)
            best_trajectory_prob += np.log(actions_distribution[state_transition.get_action()] + 1e-10)

        logger.debug("Worst Trajectory reward: %f", self.trajectories[len(self.trajectories) - 1][0])
        logger.debug("Best Trajectory reward: %f", self.trajectories[0][0])
        return best_trajectory_prob

    def generate_new_trajectory(self):
        logger.debug("Generating new trajectory")
        max_steps = props.getint('train', 'max_steps')
        step_size = props.getint('train', 'step_size')

        # perform a rollout
        state = env.reset()
        terminal_reached = False
        steps = 0
        total_reward = 0
        new_trajectory = []
        while not terminal_reached and steps < max_steps:
            state_features = self.agent.feature.phi(state)
            # get recommended action and the action distribution using policy
            action, actions_distribution = self.agent.get_policy().get_action(state_features)
            next_state, reward, done, info = env.step(action)

            for x in range(step_size - 1):
                if done:
                    terminal_reached = True
                    break
                next_state, reward2, done, info = env.step(action)
                reward += reward2

            # insert state transition to the trajectory
            state_transition = StateTransition(state, action, reward, next_state)
            new_trajectory.append(state_transition)
            total_reward += reward
            state = next_state
            steps += 1
            if done:
                terminal_reached = True

        logger.debug("Finished: Generating new trajectory")
        return total_reward, new_trajectory


def test_agent(agent, iteration_count):
    t_start = datetime.now()

    test_episodes = props.getint('test', 'test_episodes')
    max_steps = props.getint('test', 'max_steps')
    step_size = props.getint('test', 'step_size')

    total_steps = 0.0
    total_rewards = 0.0
    # Perform testing in parallel

    for i in range(test_episodes):
        state = env.reset()
        terminal_reached = False
        steps = 0
        while not terminal_reached and steps < max_steps:
            if display_game:
                env.render()
            state_features = agent.feature.phi(state)
            action, actions_distribution = agent.get_policy().get_action(state_features)
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

    # initialise experiment
    pool = Pool(processes=props.getint('multiprocess', 'num_processes'))
    experiment = EM(pool)

    display_game = True if args.display == 'true' else False
    try:
        experiment.execute_algorithm()
    except KeyboardInterrupt:
        logger.debug("User break.")
    finally:
        pool.terminate()
        env.close()

    # Upload to the scoreboard. We could also do this from another
    # logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    # gym.upload(outdir)
