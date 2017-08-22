import random
import numpy as np
import math
import logging
from datetime import datetime
import scipy.stats as stats

logger = logging.getLogger()


class SoftmaxPolicy(object):
    def __init__(self, dimension, num_actions, feature):
        self.dimension = dimension
        self.feature = feature
        self.num_actions = num_actions
        self.sigma = 1.0
        self.default_learning_rate = 0.01
        self.kl_threshold = 0.1
        self.tiny = 1e-8
        self.initialise_parameters()

    def get_policy_parameters(self):
        return np.copy(self.parameters)

    def set_policy_parameters(self, parameters, dimension=1):
        self.parameters = parameters

    def initialise_parameters(self):
        """
        TODO: See different ways of initialising the parameters.
         - Zero vectors
         - Random vectors (capped to [-10, 10] for example)
         - Maximising log likelihood etc
        :return:
        """
        # self.parameters = np.random.uniform(low=self.tiny, high=1, size=(self.num_actions, self.dimension))
        self.parameters = np.zeros(shape=(self.num_actions * self.dimension), dtype=float)
        # self.parameters.fill(self.tiny)

    def get_num_actions(self):
        return self.num_actions

    def get_action(self, state_feature):
        '''
        Perform dot product between state feature and policy parameter and return sample from the normal distribution
        :param state_feature:
        :return:
        '''

        # for each policy parameter (representing each action)
        # calculate phi /cdot theta
        # put these into array and softmax and compute random sample
        action_probabilities = []
        policy_parameters = np.split(self.parameters, self.num_actions)
        for i, parameter in enumerate(policy_parameters):
            mu = np.dot(state_feature, parameter)
            action_probabilities.append(mu)

        # substract the largest value of actions to avoid erroring out when trying to find exp(value)
        max_value = action_probabilities[np.argmax(action_probabilities)]
        for i in range(len(action_probabilities)):
            action_probabilities[i] = action_probabilities[i] - max_value

        softmax = np.exp(action_probabilities) / np.sum(np.exp(action_probabilities), axis=0)

        p = random.uniform(0, 1)
        if p < 0.05:
            chosen_policy_index = random.randint(0, len(softmax) - 1)
        else:
            chosen_policy_index = np.argmax(softmax)

        return chosen_policy_index, softmax

    def dlogpi(self, state_feature, action):
        """
        Add delta to policy parameters. one component at a time.
        Then calculcate the probability of producing the action

        :param state:
        :param action:
        :return:
        """
        _, pi = self.get_action(state_feature)

        dlogpi_parameters = np.empty(self.num_actions, dtype=object)
        # for the theta parameter used for action (use index)
        for i in range(self.num_actions):
            if i == action:
                dlogpi_parameters[i] = np.dot((1 - pi[action]), state_feature)
            else:
                theta_x = self.parameters[self.dimension * i: self.dimension * (i + 1)]
                theta_action = self.parameters[self.dimension * action: self.dimension * (action + 1)]
                component1 = -1.0 * pi[action] * (np.exp(np.dot(theta_x, state_feature))/np.exp(np.dot(theta_action, state_feature)))
                dlogpi_parameters[i] = np.dot(component1, state_feature)

        return np.concatenate(dlogpi_parameters)

    def update_parameters(self, d_error_squared, state_transitions):
        current_policy_parameters = np.copy(self.parameters)

        new_policy_parameters = self.__calculate_new_parameters(current_policy_parameters, d_error_squared)

        self.set_policy_parameters(new_policy_parameters)
        # Perform KL Divergence check
        # learning_rate = self.default_learning_rate
        # for j in range(5):
        #     kl_difference = self.avg_kl_divergence(state_transitions, new_policy_parameters, current_policy_parameters)
        #     if kl_difference < self.kl_threshold:
        #         self.set_policy_parameters(new_policy_parameters)
        #         break
        #     else:
        #         logger.debug("Not updating policy parameter as kl_difference was %f. Learning rate=%f", kl_difference,
        #                      learning_rate)
        #         learning_rate /= 10  # reduce learning rate
        #         # recalculate gradient using the new learning rate
        #         new_policy_parameters = self.__calculate_new_parameters(current_policy_parameters, d_error_squared,
        #                                                                 learning_rate=learning_rate)

    def __calculate_new_parameters(self, current_parameters, delta_vector, learning_rate=None):
        new_parameter = np.zeros(shape=len(current_parameters), dtype=float)

        if learning_rate is None:
            learning_rate = self.default_learning_rate

        for i, param in enumerate(current_parameters):
            new_parameter[i] = max(min(param - learning_rate * delta_vector[i], 10), -10)

        return new_parameter

    def avg_kl_divergence(self, state_transitions, new_policy_parameters, old_policy_parameters):
        """
        S = sum(pk * log(pk / qk), axis=0)
        :return:

        for each starting_state in state_transitions:
            * Calculate the probability of actions using old policy parameter
            * Calculate the probability of actions using new policy parameter
            * Calculate KL-Divergence for state
            * Add both to sum
        divide sum by num of states
        return average KL-Divergence
        """
        kl_sum = 0
        for state_transition in state_transitions:
            self.set_policy_parameters(new_policy_parameters)
            _, new_action_distribution = self.get_action(self.feature.phi(state_transition.get_start_state()))
            self.set_policy_parameters(old_policy_parameters)
            _, old_action_distribution = self.get_action(self.feature.phi(state_transition.get_start_state()))
            kl_sum += stats.entropy(new_action_distribution, old_action_distribution)

        return kl_sum/len(state_transitions)
