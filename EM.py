from __future__ import print_function

import multiprocessing
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


class NeatEM(object):
    def __init__(self):
        self.trajectories = []
        self.__initialise_trajectories(props.getint('initialisation', 'trajectory_size'))
        self.__initialise_agents()
        self.best_agents = []

    def __initialise_agents(self):
        self.agent = NeatEMAgent(props.getint('neuralnet', 'dimension'),
                                 props.getint('policy', 'num_actions'))

    def __initialise_trajectories(self, num_trajectories):
        '''
        Initialise trajectories of size: 'size'.
        Each trajectory we store N number of state transitions. (state, reward, next state, action)
        :param num_trajectories:
        :return:
        '''
        logger.debug("Initialising trajectories")
        t_start = datetime.now()

        max_steps = props.getint('initialisation', 'max_steps')
        step_size = props.getint('initialisation', 'step_size')
        for i in range(num_trajectories):
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

            # we have to insert timestamp as second entry so that we can order trajectories with the same reward count
            heapq.heappush(self.trajectories, (total_reward, datetime.now(), trajectory))
        logger.debug("Finished: Creating trajectories. Time taken: %f", (datetime.now() - t_start).total_seconds())

    def execute_algorithm(self, generations):
        for i in range(generations):
            self.fitness_function()

    def fitness_function(self):
        """
        Generate trajectory.
        Insert into Trajectories.
        Select best trajectory and perform policy update 
        :return: 
        """
        tstart = datetime.now()

        total_reward, state_transitions = self.generate_new_trajectory()

        heapq.heappush(self.trajectories, (total_reward, datetime.now(), state_transitions))

        # '''
        # For each individual update value function using TD error and experience replay
        # And update the policy parameter
        # '''
        # experience_replay = props.getint('evaluation', 'experience_replay')
        # random_state_transitions = random.sample(state_transitions, experience_replay)

        # strip weak trajectories from trajectory_set
        self.trajectories = heapq.nlargest(props.getint('initialisation', 'trajectory_size'), self.trajectories)

        # now assign fitness to each individual/genome
        # fitness is the log prob of following the best trajectory
        best_trajectory = self.trajectories[0]

        # update value function
        self.agent.update_value_function(best_trajectory[2])

        # update policy parameter
        self.agent.update_policy_function(best_trajectory[2])

        best_trajectory_prob = 0
        total_reward, _, trajectory_state_transitions = best_trajectory
        for j, state_transition in enumerate(trajectory_state_transitions):
            # calculate probability of the action where policy action = action
            state_features = self.agent.feature.phi(state_transition.get_start_state())
            _, actions_distribution = self.agent.get_policy().get_action(state_features)
            best_trajectory_prob += np.log(actions_distribution[state_transition.get_action()])

        fitness = best_trajectory_prob
        self.agent.fitness = fitness

        logger.debug("Worst Trajectory reward: %f", self.trajectories[len(self.trajectories) - 1][0])
        logger.debug("Best Trajectory reward: %f", self.trajectories[0][0])

        # save the best individual's genome
        logger.debug("Agent fitness: %f", self.agent.fitness)

        logger.debug("Completed Generation. Time taken: %f", (datetime.now() - tstart).total_seconds())

    def generate_new_trajectory(self):
        logger.debug("Generating %d new trajectory")
        max_steps = props.getint('initialisation', 'max_steps')
        step_size = props.getint('initialisation', 'step_size')

        # perform a rollout
        state = env.reset()
        terminal_reached = False
        steps = 0
        total_reward = 0
        new_trajectory = []
        while not terminal_reached and steps < max_steps:
            # env.render()
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

        logger.debug("Finished: Generating new trajectories")
        return total_reward, new_trajectory


def test_best_agent(agent):
    t_start = datetime.now()

    max_steps = props.getint('initialisation', 'max_steps')
    test_episodes = props.getint('test', 'test_episodes')
    step_size = props.getint('initialisation', 'step_size')

    total_steps = 0.0
    total_rewards = 0.0
    for i in range(test_episodes):
        state = env.reset()
        terminal_reached = False
        steps = 0
        while not terminal_reached and steps < max_steps:
            env.render()
            state_features = agent.feature.phi(state)
            action, actions_distribution = agent.get_policy().get_action(state_features)
            state, reward, done, info = env.step(action)

            for x in range(step_size - 1):
                if done:
                    terminal_reached = True
                    break
                state, reward2, done, info = env.step(action)
                reward += reward2

            total_rewards += reward

            steps += 1
            if done:
                terminal_reached = True
        total_steps += steps
    average_steps_per_episodes = total_steps / test_episodes
    average_rewards_per_episodes = total_rewards / test_episodes

    # save this to file along with the generation number
    entry = [average_steps_per_episodes, average_rewards_per_episodes]
    with open(r'agent_evaluation.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(entry)

    logger.debug("Finished: evaluating best agent. Time taken: %f", (datetime.now() - t_start).total_seconds())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    args = parser.parse_args()

    gym.undo_logger_setup()
    logging.basicConfig(filename='log/debug-'+str(datetime.now())+'.log', level=logging.DEBUG)
    logger = logging.getLogger()
    logging.Formatter('[%(asctime)s] %(message)s')
    env = gym.make(args.env_id)

    # logger.debug(env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'))
    # env.spec.tags['wrapper_config.TimeLimit.max_episode_steps'] = 200
    # logger.debug(env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'))
    # env._max_episode_steps = 200

    # env = env.env
    logger.debug("action space: %s", env.action_space)
    logger.debug("observation space: %s", env.observation_space)


    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    # outdir = '~/tmp/neat-em-data/' + str(datetime.now())
    # env = wrappers.Monitor(env, directory=outdir, force=True)
    local_dir = os.path.dirname(__file__)
    # load properties
    logger.debug("Loading Properties File")
    props = configparser.ConfigParser()
    prop_path = os.path.join(local_dir, 'properties/{0}/neatem_properties.ini'.format(env.spec.id))
    props.read(prop_path)
    logger.debug("Finished: Loading Properties File")

    # Load the config file, which is assumed to live in
    # the same directory as this script.

    config_path = os.path.join(local_dir, 'properties/{0}/config'.format(env.spec.id))
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    population = NeatEM()

    # Run until the winner from a generation is able to solve the environment
    # or the user interrupts the process.
    try:
        # Run for 5 generations
        population.execute_algorithm(props.getint('neuralnet', 'generation'))

        # Generate test results
        outdir = 'videos/tmp/neat-em-data/{0}-{1}'.format(env.spec.id, str(datetime.now()))
        env = wrappers.Monitor(env, directory=outdir, force=True)
        test_best_agent(population.agent)

        # visualize.plot_stats(population.stats, ylog=False, view=False, filename="fitness.svg")

        # mfs = sum(population.stats.get_fitness_mean()[-20:]) / 20.0
        # logger.debug("Average mean fitness over last 20 generations: %f", mfs)

        # mfs = sum(population.stats.get_fitness_stat(min)[-20:]) / 20.0
        # logger.debug("Average min fitness over last 20 generations: %f", mfs)

        # Use the 10 best genomes seen so far
        # best_genomes = population.stats.best_unique_genomes(10)

        # save_best_genomes(best_genomes, True)

    except KeyboardInterrupt:
        logger.debug("User break.")
        # save the best neural network or save top 5?
        # best_genomes = population.stats.best_unique_genomes(5)

        # save_best_genomes(best_genomes, False)

    env.close()

    # Upload to the scoreboard. We could also do this from another
    # logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    # gym.upload(outdir)
