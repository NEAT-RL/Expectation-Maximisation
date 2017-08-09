from ValueFunction.ValueFunction import ValueFunction
from Policy.SoftmaxPolicy import SoftmaxPolicy
import numpy as np
import Feature
import math
"""
Mountain car
state = (position, velocity)
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        min speed = -self.max_speed = -0.07
         
TODO: 
Add Cartpole settings
"""


class NeatEMAgent(object):
    def __init__(self, dimension, num_actions):
        self.dimension = dimension
        # self.feature = self.__create_discretised_feature([5, 5], 2, [-1.2, -0.07], [0.6, 0.07])
        self.feature = self.__create_discretised_feature([10, 10, 10, 10], 4, [-2.4, -10, -41.8 * math.pi / 180, -10], [2.4, 10, 41.8 * math.pi / 180, 10])
        self.valueFunction = ValueFunction(dimension)
        self.policy = SoftmaxPolicy(dimension, num_actions)
        self.fitness = 0
        self.gamma = 0.99

    @staticmethod
    def __create_discretised_feature(partition_size, state_length, state_lower_bounds, state_upper_bounds):
        """
        :param partition_size: Array of partition_sizes for each state field
        :param state_length: Number of states == input dimension (state)
        :param state_lower_bounds: array of lower bounds for each state field
        :param state_upper_bounds: array of upper bounds for each state field
        :return: discretised feature
        """
        intervals = []
        output_dimension = 0

        for i in range(state_length):
            output_dimension += partition_size[i]
            state_col = Feature.DiscretizedFeature.create_partition(state_lower_bounds[i],
                                                                    state_upper_bounds[i], partition_size[i])
            intervals.append(state_col)
        return Feature.DiscretizedFeature(state_length, output_dimension, intervals)

    def get_feature(self):
        return self.feature

    def get_value_function(self):
        return self.valueFunction

    def get_policy(self):
        return self.policy

    def get_fitness(self):
        return self.fitness

    def update_value_function(self, state_transitions):
        delta_omega = np.zeros(self.dimension, dtype=float)

        for state_transition in state_transitions:
            old_state_features = self.feature.phi(state_transition.get_start_state())
            new_state_features = self.feature.phi(state_transition.get_end_state())
            derivative = 2 * (self.valueFunction.get_value(old_state_features) - (state_transition.get_reward() + self.gamma * self.valueFunction.get_value(new_state_features)))
            delta = np.dot(derivative, self.valueFunction.get_parameter())
            delta_omega += delta

        delta_omega /= len(state_transitions)
        self.valueFunction.update_parameters(delta_omega)

    def update_policy_function(self, state_transitions):
        """
        Need to update the policy parameters which was used to make the action.
        There are N number of discrete actions.
        This N number is stored in Policy so I should be looking to put this method there 
        :param state_transitions:
        :return:
        """
        # Update the policy parameters for the actions that are taken

        # Create a delta vector of size [# actions] where each element is a delta policy object
        delta = np.array([DeltaPolicy(self.dimension) for _ in range(self.policy.get_num_actions())])

        for state_transition in state_transitions:
            phi = self.feature.phi(state_transition.get_start_state())
            _, actions_distribution = self.policy.get_action(phi)

            action_prob = actions_distribution[state_transition.get_action()]
            # check for matrix multiplication
            phi_dot = np.dot(action_prob * (action_prob - 1), np.array(phi))
            # convert vector into a column vector
            phi_dot = phi_dot.reshape(-1, 1)

            component1 = np.matmul(phi_dot, phi_dot.transpose())  # produces a dimension * dimension matrix

            phi_new = self.feature.phi(state_transition.get_end_state())
            td_error = state_transition.get_reward() + self.gamma * self.valueFunction.get_value(phi_new) - self.valueFunction.get_value(phi)

            component2 = np.dot((1 - action_prob) * td_error, phi)

            # we only update the policy parameter that was used to perform the action
            delta[state_transition.get_action()].add(component1, component2)

        for i in range(len(delta)):
            delta[i].component1 = delta[i].component1 / 1  # delta[i].state_transition_count
            delta[i].component2 = np.dot(2.0 / 1, delta[i].component2)
            delta[i].calculate_delta()

        self.policy.update_parameters(delta)  # delta is a vector of size (num of actions) and each element is a vector of policy parameter


class DeltaPolicy(object):

    def __init__(self, dimension):
        self.component1 = 0
        self.component2 = np.zeros(dimension, dtype=float)
        self.delta = np.zeros(dimension, dtype=float)
        self.state_transition_count = 0.0

    def add(self, component1, component2):
        self.component1 += component1
        self.component2 += component2
        self.state_transition_count += 1.0

    def calculate_delta(self):
        self.delta = np.dot(self.component1, self.component2)

