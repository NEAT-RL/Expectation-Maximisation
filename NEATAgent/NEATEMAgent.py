from ValueFunction.ValueFunction import ValueFunction
from Policy.SoftmaxPolicy import SoftmaxPolicy
import numpy as np
from . import Feature
import math
from multiprocessing import Pool
import theano
import logging
import theano.tensor as T
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
logger = logging.getLogger()


class NeatEMAgent(object):
    def __init__(self, dimension, num_actions, state_length, feature_size):
        self.dimension = dimension
        self.num_actions = num_actions
        # self.feature = Feature.discretizedFeature(state_length, feature_size)
        # Cartpole-v0
        self.feature = self.__create_discretised_feature([10, 10, 10, 10],
                                                         state_length,
                                                         [-2.4, -10, -41.8 * math.pi / 180, -10],
                                                         [2.4, 10, 41.8 * math.pi / 180, 10])
        self.valueFunction = ValueFunction(dimension)
        self.policy = SoftmaxPolicy(dimension, num_actions, self.feature)
        self.fitness = 0
        self.gamma = 0.99
        self.best_policy_parameters = None
        self.best_average_reward = 0

        self.phi = T.fmatrix('phi')
        self.action = T.imatrix('action')
        self.phi_new = T.fmatrix('phi_new')
        self.reward = T.dvector('reward')
        self.theta = theano.shared(self.policy.get_policy_parameters(), 'theta')
        self.omega = theano.shared(self.valueFunction.get_parameter(), 'omega')
        logpi = T.log(T.batched_dot(T.nnet.softmax(T.dot(self.phi, self.theta)), self.action) + 1e-20)
        td_error = self.reward + self.gamma * T.dot(self.phi_new, self.omega) - T.dot(self.phi, self.omega)
        logpi_td_error = logpi * td_error
        logpi_td_error_mean = T.sum(logpi_td_error)
        # then do derivation to get e
        e = T.grad(logpi_td_error_mean, self.theta)

        de_squared = T.sum(T.jacobian(T.sqr(e).flatten(), self.theta), axis=0)
        self.delta_policy = theano.function([self.phi, self.phi_new, self.reward, self.action], de_squared)
        # Fitness function of policy
        fitness_function = T.sum(T.log(T.batched_dot(T.nnet.softmax(T.dot(self.phi, self.theta)), self.action) + 1e-20))
        self.calculate_fitness = theano.function([self.phi, self.action], fitness_function)

        # MSE for value function
        derivative_mse = T.grad(T.mean((T.dot(self.phi, self.omega) - (self.reward + self.gamma * T.dot(self.phi_new, self.omega))) ** 2), self.omega)
        self.mse_value_function = theano.function([self.phi, self.phi_new, self.reward], derivative_mse)

        self.td_error_function = theano.function([self.phi, self.phi_new, self.reward], td_error)

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

    def save_policy_parameters(self, average_reward):
        self.best_policy_parameters = self.policy.get_policy_parameters()
        self.best_average_reward = average_reward

    def update_parameters(self, all_state_starts, all_state_ends, all_actions, all_rewards, random_indexes):
        phi = []
        phi_new = []

        for i in range(len(all_state_starts)):
            phi.append(self.feature.phi(all_state_starts[i]))
            phi_new.append(self.feature.phi(all_state_ends[i]))

        value_function_phi = []
        value_function_phi_new = []
        value_function_rewards = []
        for index in random_indexes:
            value_function_phi.append(phi[index])
            value_function_phi_new.append(phi_new[index])
            value_function_rewards.append(all_rewards[index])

        # update value function
        logger.debug("Updating value function")
        self.update_value_function_theano(value_function_phi, value_function_phi_new, value_function_rewards)

        # update policy parameter
        logger.debug("Updating policy function")

        self.update_policy_function_theano(phi, phi_new, all_actions, all_rewards)

    def update_value_function(self, start_states, end_states, rewards):
        delta_omega = np.zeros(self.dimension, dtype=float)

        for i in range(len(start_states)):
            old_state_features = self.feature.phi(start_states[i])
            new_state_features = self.feature.phi(end_states[i])
            derivative = 2 * (self.valueFunction.get_value(old_state_features) - (rewards[i] + self.gamma * self.valueFunction.get_value(new_state_features)))
            delta = np.dot(derivative, self.valueFunction.get_parameter())
            delta_omega += delta

        delta_omega /= len(start_states)
        self.valueFunction.update_parameters(delta_omega)
        self.omega.set_value(self.valueFunction.get_parameter())

    def update_value_function_theano(self, start_states, end_states, rewards):
        delta = self.mse_value_function(start_states, end_states, rewards)
        self.valueFunction.update_parameters(delta)
        self.omega.set_value(self.valueFunction.get_parameter())

    def update_policy_function_theano(self, phi, phi_new, all_actions, all_rewards):
        delta_policy = self.delta_policy(phi, phi_new, all_rewards, all_actions)
        print(np.mean(self.td_error_function(phi, phi_new, all_rewards)))
        self.policy.update_parameters_theano(delta_policy, phi)
        self.theta.set_value(self.policy.get_policy_parameters())

    def update_policy_function(self, random_state_transitions, all_state_transitions, pool):
        """
        For each parameter in error squared function:
           e(theta + delta)Transpose cdot e(theta+delta) - e(theta-delta)/(2*delta)
        """
        # first copy the policy parameters
        original_policy_parameters = self.policy.get_policy_parameters()
        # Perform Parallel computation of numeric approximation of error squared function
        results = [pool.apply_async(
            self.approximate_d_error_squared, args=(x, random_state_transitions, original_policy_parameters))
                   for x in range(len(original_policy_parameters))]
        results2 = [i.get() for i in results]
        d_error_squared = [value[1] for value in sorted(results2)]  # collect the error derivative in sorted order based on index

        # set policy parameter to its original value in case it has been changed. WHICH IT SHOULDN'T BE.
        self.policy.set_policy_parameters(original_policy_parameters)
        # update policy parameter
        self.policy.update_parameters(d_error_squared, all_state_transitions)

    def calculate_agent_fitness(self, start_states, actions):
        phi = []
        for i in range(len(start_states)):
            phi.append(self.feature.phi(start_states[i]))

        self.fitness = self.calculate_fitness(phi, actions)
        return self.fitness

    def approximate_d_error_squared(self, index, random_state_transitions, original_policy_parameters):
        delta = 0.01
        # make new policy with original_policy_parameters
        policy = SoftmaxPolicy(self.dimension, self.num_actions, self.feature, self.policy.is_greedy)  # feature is not needed if KL divergence is unused
        policy.set_policy_parameters(original_policy_parameters)

        # maintain positive delta and negative delta error functions
        error_func_positive_delta = np.zeros(shape=(policy.num_actions * policy.dimension), dtype=float)
        error_func_negative_delta = np.zeros(shape=(policy.num_actions * policy.dimension), dtype=float)

        policy_parameters_positive_delta = np.copy(original_policy_parameters)
        policy_parameters_negative_delta = np.copy(original_policy_parameters)

        # add/subtract delta to both positive delta and negative delta function parameters
        policy_parameters_positive_delta[index] = policy_parameters_positive_delta[index] + delta
        policy_parameters_negative_delta[index] = policy_parameters_negative_delta[index] - delta

        # calculate the error function
        for state_transition in random_state_transitions:
            phi_start = self.feature.phi(state_transition.get_start_state())

            # set theta + delta and calculate dlogpi for positive delta case
            policy.set_policy_parameters(policy_parameters_positive_delta)
            dlogpi_positive_delta = policy.dlogpi(phi_start, state_transition.get_action())

            # set theta - delta and calculate dlogpi for negative delta case
            policy.set_policy_parameters(policy_parameters_negative_delta)
            dlogpi_negative_delta = policy.dlogpi(phi_start, state_transition.get_action())

            # calculate td_error. TD Error calculate is independent of policy
            phi_end = self.feature.phi(state_transition.get_end_state())
            td_error = state_transition.get_reward() + self.gamma * self.valueFunction.get_value(
                phi_end) - self.valueFunction.get_value(phi_start)

            # Multiply dlogpi with td error for positive delta and negative delta functions
            dlogpi_positive_delta *= td_error
            dlogpi_negative_delta *= td_error

            # added to error function positive and to error function negative cases
            error_func_positive_delta += dlogpi_positive_delta
            error_func_negative_delta += dlogpi_negative_delta

        error_func_positive_delta /= len(random_state_transitions)
        error_func_negative_delta /= len(random_state_transitions)

        '''
        now calculate scalar approximation.
        e(theta + delta) <==> error_func_negative_delta
        e(theta - delta) <==> error_func_negative_delta

        e(theta + delta)^Transpose dot e(theta+delta) - e(theta-delta)^Transpose dot e(theta-delta)/(2*delta)
        '''
        error_derivative = np.dot(np.transpose(error_func_positive_delta), error_func_positive_delta) - \
                           np.dot(np.transpose(error_func_negative_delta), error_func_negative_delta)

        error_derivative /= (2 * delta)

        return index, error_derivative
