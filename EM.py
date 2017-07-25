from __future__ import print_function

import multiprocessing
import os
import pickle
import argparse
import logging
import random
import sys
import gym
import gym.wrappers as wrappers
import configparser
import visualize
from NEATAgent.NEATEMAgent import NeatEMAgent
import numpy as np
import heapq
from datetime import datetime


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


class ExpectationMaximisation(object):
    def __init__(self):
        self.trajectories = []
        self.initialise_trajectories(props.getint('initialisation', 'trajectory_size'))
        self.generation_count = 0
        self.agents = []
        self.initialise_agents()

    def initialise_trajectories(self, num_trajectories):
        '''
        Initialise trajectories of size: 'size'.
        Each trajectory we store N number of state transitions. (state, reward, next state, action)
        :param num_trajectories:
        :return:
        '''
        logger.debug("Initialising trajectories")
        tstart = datetime.now()

        max_steps = props.getint('initialisation', 'max_steps')
        for i in range(num_trajectories):
            trajectory = []
            state = env.reset()
            terminal_reached = False
            steps = 0
            reward_count = 0
            while not terminal_reached and steps < max_steps:
                # sample action from the environment
                action = env.action_space.sample()
                next_state, reward, done, info = env.step(action)
                state_transition = StateTransition(state, action, reward, next_state)
                # insert state transition to the trajectory
                trajectory.append(state_transition)
                reward_count += reward
                state = next_state
                if done:
                    terminal_reached = True
                steps += 1

            '''
            we have to insert timestamp as second entry in tuple so that we can order trajectories with 
            the same reward count.
            The self.trajectories is a priority queue of (total_reward, time_stamp, state_transitions)
            The self.trajectories orders the tuples by total_reward count. 
                If total_reward is same then it orders by timestamp
            '''
            heapq.heappush(self.trajectories, (reward_count, datetime.now(), trajectory))
        logger.debug("Finished: Initialising trajectories. Time taken: %f", (datetime.now() - tstart).total_seconds())

    def initialise_agents(self):
        logger.debug("Initialising agents")
        num_agents = props.getint('initialisation', 'num_agents')

        for i in range(num_agents):
            # reinitialise the agents
            agent = NeatEMAgent(props.getint('neuralnet', 'dimension'),
                                props.getint('policy', 'num_actions'))
            self.agents.append(agent)

        logger.debug("Finished: Initialising neural networks")

    def reset_agent_fitness(self):
        for i in range(len(self.agents)):
            self.agents[i].reset_fitness()

    def execute_algorithm(self, generations):
        for i in range(generations):
            self.evaluate_algorithm()

    def evaluate_algorithm(self):
        """
        This method is called every generation.
        Create new array
        :return:
        """

        '''
        Collect a set of state_transitions from the trajectories
        '''
        state_transitions = set()
        for i in range(len(self.trajectories)):
            _, __, trajectory = self.trajectories[i]
            state_transitions = state_transitions | set(trajectory)

        tstart = datetime.now()
        '''
        For each individual update value function using TD error and experience replay
        And update the policy parameter
        '''
        for agent in self.agents:
            experience_replay = props.getint('evaluation', 'experience_replay')
            random_state_transitions = random.sample(state_transitions, experience_replay)
            for i in range(experience_replay):
                state_transition = random_state_transitions[i]
                # update TD error and value function
                agent.update_value_function(state_transition.get_start_state(), state_transition.get_end_state(),
                                            state_transition.get_reward())

            # update policy parameter
            agent.update_policy_function(state_transitions)

            # now assign fitness to each individual/genome
            # fitness is the log prob of following the best trajectory
            # I need the get action to return me the probabilities of the actions rather than a numerical action
            best_trajectory = self.trajectories[len(self.trajectories) - 1]
            best_trajectory_prob = 0
            total_reward, _, trajectory_state_transitions = best_trajectory
            for j, state_transition in enumerate(trajectory_state_transitions):
                # calculate probability of the action probability where policy action = action
                state_features = agent.get_feature().phi(state_transition.get_start_state())
                _, actions_distribution = agent.get_policy().get_action(state_features)
                best_trajectory_prob += np.log(actions_distribution[state_transition.get_action()])

            agent.fitness = best_trajectory_prob

        # select K random agents to perform rollout

        num_new_trajectories = props.getint('evaluation', 'new_trajectories')

        logger.debug("Generating %d new trajectories", num_new_trajectories)

        max_steps = props.getint('initialisation', 'max_steps')
        rand_policy = random.randint(0, len(self.agents) - 1)
        for i in range(num_new_trajectories):
            agent = self.agents[rand_policy]
            # perform a rollout
            state = env.reset()
            terminal_reached = False
            steps = 0
            reward_count = 0
            new_trajectory = []
            while not terminal_reached and steps < max_steps:
                # env.render()
                state_features = agent.get_feature().phi(state)
                action, actions_distribution = agent.get_policy().get_action(state_features)
                next_state, reward, done, info = env.step(action)
                # insert state transition to the trajectory
                state_transition = StateTransition(state, action, reward, next_state)
                new_trajectory.append(state_transition)
                reward_count += reward
                state = next_state
                if done:
                    terminal_reached = True
                steps += 1

            '''
            In MountainCar problem, the total reward is always negative. The lower the number the worse the trajectory
            Heapq keeps smallest values in the first position.
            '''
            heapq.heappush(self.trajectories, (reward_count, datetime.now(), new_trajectory))

            new_policy = random.randint(0, len(self.agents) - 1)
            while rand_policy == new_policy:
                new_policy = random.randint(0, len(self.trajectories) - 1)
            rand_policy = new_policy

        # strip weak trajectories from trajectory_set and add state transitions to set state_transitions
        self.trajectories = self.trajectories[
                            len(self.trajectories) - props.getint('initialisation', 'trajectory_size'):]

        logger.debug("Worst Trajectory reward: %f", self.trajectories[0][0])
        logger.debug("Best Trajectory reward: %f", self.trajectories[len(self.trajectories) - 1][0])

        # save the best individual's genome. Best genome is the one with the smallest fitness value
        agent = max(self.agents, key=lambda x: x.fitness)
        logger.debug("Best agent fitness: %f", agent.fitness)

        agent = min(self.agents, key=lambda x: x.fitness)
        logger.debug("Worst agent fitness: %f", agent.fitness)
        # save genome

        self.generation_count += 1
        logger.debug("Completed Generation %d. Time taken: %f", self.generation_count, (datetime.now() - tstart).total_seconds())

        self.reset_agent_fitness()


def save_best_genomes(best_genomes, has_won, config):

    for n, g in enumerate(best_genomes):
        name = "results/"
        if has_won:
            name += 'winner-{0}'.format(n)
        else:
            name += 'best-{0}'.format(n)

        with open(name + '.pickle', 'wb') as f:
            pickle.dump(g, f)

        visualize.draw_net(config, g, view=False, filename=name + "-net.gv")
        visualize.draw_net(config, g, view=False, filename=name + "-net-enabled.gv",
                           show_disabled=False)
        visualize.draw_net(config, g, view=False, filename=name + "-net-enabled-pruned.gv",
                           show_disabled=False, prune_unused=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='MountainCar-v0', help='Select the environment to run')
    args = parser.parse_args()

    gym.undo_logger_setup()
    logging.basicConfig(filename='log/debug-'+str(datetime.now())+'.log', level=logging.DEBUG)
    logger = logging.getLogger()
    logging.Formatter('[%(asctime)s] %(message)s')
    env = gym.make(args.env_id).env

    logger.debug("action space: %s", env.action_space)
    logger.debug("observation space: %s", env.observation_space)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    # outdir = '/tmp/neat-em-data/' + str(datetime.now())
    # env = wrappers.Monitor(env, directory=outdir, force=True)

    # load properties
    logger.debug("Loading Properties File")
    props = configparser.ConfigParser()
    props.read('neatem_properties.ini')
    logger.debug("Finished: Loading Properties File")

    population = ExpectationMaximisation()

    # Run until the winner from a generation is able to solve the environment
    # or the user interrupts the process.
    while 1:
        try:
            # Run for 5 generations
            population.execute_algorithm(props.getint('neuralnet', 'generation'))

            # visualize.plot_stats(population.stats, ylog=False, view=False, filename="fitness.svg")
            #
            # mfs = sum(population.stats.get_fitness_mean()[-20:]) / 20.0
            # logger.debug("Average mean fitness over last 20 generations: %f", mfs)
            #
            # mfs = sum(population.stats.get_fitness_stat(min)[-20:]) / 20.0
            # logger.debug("Average min fitness over last 20 generations: %f", mfs)
            #
            # # Use the 10 best genomes seen so far
            # best_genomes = population.stats.best_unique_genomes(10)
            #
            # save_best_genomes(best_genomes, True, config)
            # break

        except KeyboardInterrupt:
            logger.debug("User break.")
            # save the best neural network or save top 5?
            # best_genomes = population.stats.best_unique_genomes(5)

            # save_best_genomes(best_genomes, False, config)
            break

    env.close()

    # Upload to the scoreboard. We could also do this from another
    # logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    # gym.upload(outdir)
