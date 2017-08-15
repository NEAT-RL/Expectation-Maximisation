from ValueFunction.ValueFunction import ValueFunction
from Policy.SoftmaxPolicy import SoftmaxPolicy
import numpy as np
from . import Feature
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
        self.feature = self.__create_discretised_feature(4, 16)
        self.valueFunction = ValueFunction(dimension)
        self.policy = SoftmaxPolicy(dimension, num_actions, self.feature)
        self.fitness = 0
        self.gamma = 0.99

    @staticmethod
    def __create_discretised_feature(state_length, output_dimension):
        """
        :param partition_size: Array of partition_sizes for each state field
        :param state_length: Number of states == input dimension (state)
        :param state_lower_bounds: array of lower bounds for each state field
        :param state_upper_bounds: array of upper bounds for each state field
        :return: discretised feature
        """
        return Feature.DiscretizedFeature(state_length, output_dimension)

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

    def update_policy_function(self, trajectories, state_transitions):
        derror_squared = self.__d_error_squared(trajectories)
        # update policy
        self.policy.update_parameters(derror_squared, state_transitions)

    def __d_error_squared(self, trajectories):
        delta = 0.001
        # Create a derivative_of_error_squared vector. We calculate derivative w.r.t. theta
        d_error_squared = np.zeros(shape=(self.policy.num_actions * self.dimension), dtype=float)

        '''
        For each parameter in error squared function:
           e(theta + delta)Transpose cdot e(theta+delta) - e(theta-delta)/(2*delta)
        '''
        # first copy the policy parameters
        original_policy_parameters = self.policy.get_policy_parameters()

        for i in range(len(original_policy_parameters)):
            error_func_positive_delta = np.zeros(shape=(self.policy.num_actions * self.dimension), dtype=float)
            error_func_negative_delta = np.zeros(shape=(self.policy.num_actions * self.dimension), dtype=float)

            new_parameters_positive_delta = np.copy(original_policy_parameters)
            new_parameters_negative_delta = np.copy(original_policy_parameters)

            new_parameters_positive_delta[i] = new_parameters_positive_delta[i] + delta
            new_parameters_negative_delta[i] = new_parameters_negative_delta[i] - delta

            for j, (_, __, state_transitions) in enumerate(trajectories):
                for state_transition in state_transitions:
                    phi_old = self.feature.phi(state_transition.get_start_state())

                    # set theta + delta and calculate dlogpi_positive_delta
                    self.policy.set_policy_parameters(new_parameters_positive_delta)
                    dlogpi_positive_delta = self.policy.dlogpi(phi_old, state_transition.get_action())

                    # set theta - delta and calculate dlogpi_negative_delta
                    self.policy.set_policy_parameters(new_parameters_negative_delta)
                    dlogpi_negative_delta = self.policy.dlogpi(phi_old, state_transition.get_action())

                    # calculate td_error
                    phi_new = self.feature.phi(state_transition.get_end_state())
                    td_error = state_transition.get_reward() + self.gamma * self.valueFunction.get_value(
                        phi_new) - self.valueFunction.get_value(phi_old)

                    dlogpi_positive_delta *= td_error
                    dlogpi_negative_delta *= td_error

                    error_func_positive_delta += dlogpi_positive_delta
                    error_func_negative_delta += dlogpi_negative_delta

            error_func_positive_delta /= len(trajectories)
            error_func_negative_delta /= len(trajectories)

            '''
            now calculate scalar approximation
            e(theta + delta)^Transpose dot e(theta+delta) - e(theta-delta)^Transpose dot e(theta-delta)/(2*delta)
            '''
            error_derivative = np.dot(np.transpose(error_func_positive_delta), error_func_positive_delta) - np.dot(
                np.transpose(error_func_negative_delta), error_func_negative_delta)
            error_derivative /= (2 * delta)
            d_error_squared[i] = error_derivative

        self.policy.set_policy_parameters(original_policy_parameters)
        return d_error_squared


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

