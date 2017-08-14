import random
import numpy as np
import math
import logging
from datetime import datetime
import scipy.stats as stats

logging.basicConfig(filename='log/policy-debug-' + str(datetime.now()) + '.log', level=logging.DEBUG)
logger = logging.getLogger()


class SoftmaxPolicy(object):
    def __init__(self, dimension, num_actions):
        self.dimension = dimension
        self.parameters = []
        self.num_actions = num_actions
        self.sigma = 1.0
        self.default_learning_rate = 0.01
        self.kl_threshold = 0.1
        self.tiny = 1e-8
        self.initialise_parameters()

    def get_policy_parameters(self):
        return np.concatenate((self.parameters), axis=0)

    def set_policy_parameters(self, parameters, dimension=1):
        if dimension == 1:
            self.parameters = np.split(parameters, self.num_actions)
        elif dimension == self.num_actions:
            self.parameters = parameters
        else:
            raise Exception("dimension must be either 1 or number of actions")

    def initialise_parameters(self):
        """
        TODO: See different ways of initialising the parameters.
         - Zero vectors
         - Random vectors (capped to [-10, 10] for example)
         - Maximising log likelihood etc
        :return: 
        """
        self.parameters = np.random.uniform(low=self.tiny, high=1, size=(self.num_actions, self.dimension))
        # self.parameters = np.zeros(shape=(self.num_actions, self.dimension), dtype=float)
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
        for i, parameter in enumerate(self.parameters):
            mu = np.dot(state_feature, parameter)
            action_probabilities.append(mu)

        # substract the largest value of actions to avoid erroring out when trying to find exp(value)
        max_value = action_probabilities[np.argmax(action_probabilities)]
        for i in range(len(action_probabilities)):
            action_probabilities[i] = action_probabilities[i] - max_value

        softmax = np.exp(action_probabilities) / np.sum(np.exp(action_probabilities), axis=0)

        p = random.uniform(0, 1)
        if p < 0.02:
            chosen_policy_index = random.randint(0, len(softmax) - 1)
        else:
            chosen_policy_index = np.argmax(softmax)

        return chosen_policy_index, softmax

    def dlogpi(self, state_feature, action):
        """
        Add delta to all of the 3 policy parameters. one component at a time
        Then calculcate the probability of producing the action
        Then calculcate paraderivative
        
        :param state: 
        :param action: 
        :return: 
        """
        original_parameters = np.concatenate((self.parameters), axis=0)
        
        para_derivatives = np.zeros(len(original_parameters), dtype=float)
        
        for i in range(len(original_parameters)):
            new_parameters = np.copy(original_parameters)
            new_parameters[i] = new_parameters[i] - 0.00001
        
            self.parameters = np.split(new_parameters, self.num_actions)
            _, action_distribution = self.get_action(state_feature)
            prev_prob = action_distribution[action]
        
            new_parameters[i] = new_parameters[i] + 0.00001 * 2
            self.parameters = np.split(new_parameters, self.num_actions)
            _, action_distribution = self.get_action(state_feature)
            after_prob = action_distribution[action]
        
            para_derivatives[i] = (after_prob - prev_prob) / 2. / 0.00001
        
        # Replace parameters with the original parameters
        self.parameters = np.split(original_parameters, self.num_actions)
        
        return para_derivatives  # return new parameters
        '''
        recommended_action, action_distribution = self.get_action(state_feature)
        dlogpi = state_feature * (1 - action_distribution[action])

        # return array of size self.num_actions * self.dimension with padding of zeros for policy parameter not chosen as action
        return np.lib.pad(dlogpi,
                          ((action - 0) * self.dimension, (self.num_actions - 1 - action) * self.dimension),
                          'constant',
                          constant_values=(0, 0)
                          )
        '''

    def update_parameters(self, delta):
        """
        Delta is an array where each element is delta for a policy parameter.
        Note: Number of policy parameters = number of actions.
        Each delta object contains a delta of the policy parameter.
        :param delta: 
        :return:
        Assume size of delta == number of actions
        """
        # Calculate KL-divergence
        delta = np.split(delta, self.num_actions)
        new_parameters = np.zeros(shape=(self.num_actions, self.dimension), dtype=float)

        for i in range(len(self.parameters)):
            new_parameters[i] = self.__calculate_gradient(self.parameters[i], delta[i])

        for i in range(len(self.parameters)):
            learning_rate = self.default_learning_rate
            for j in range(10):
                kl_difference = self.calculate_kl_divergence(np.array(new_parameters[i]), np.array(self.parameters[i]))
                if kl_difference < self.kl_threshold:
                    self.parameters[i] = new_parameters[i]
                    break
                else:
                    logger.debug("Not updating policy parameter as kl_difference was %f. Learning rate=%f",
                                 kl_difference, learning_rate)
                    learning_rate /= 10  # reduce learning rate
                    # recalculate gradient using the new learning rate
                    new_parameters[i] = self.__calculate_gradient(self.parameters[i], delta[i], learning_rate)

    def __calculate_gradient(self, parameter, delta_vector, learning_rate=None):
        new_parameter = np.zeros(shape=self.dimension, dtype=float)

        if learning_rate is None:
            learning_rate = self.default_learning_rate

        for j, param in enumerate(parameter):
            new_parameter[j] = max(min(param - learning_rate * delta_vector[j], 10), -10)

        return new_parameter

    def calculate_kl_divergence(self, pk, qk):
        """
        S = sum(pk * log(pk / qk), axis=0)
        :return: 
        """
        pk_norm = np.array(self.normalize(pk))
        qk_norm = np.array(self.normalize(qk))

        # the normalised arrays may have negative number. So I just clip it to 0. # TODO: CHECK WITH YIMING.
        # OR SHOULD I NORMALISE THE VALUES TO BE BETWEEN [0, 1] AND THEN CALCULATE THE PROBABILITY DISTRIBUTION
        # pk_norm = pk_norm.clip(min=0)
        # pk_norm = qk_norm.clip(min=0)
        kl_sum = 0
        for i in range(len(pk)):
            if math.isnan(pk_norm[i] * np.log(math.fabs(pk_norm[i])/math.fabs((qk_norm[i] + self.tiny)) + self.tiny)):
                print("KL divergence resulted in nan. Normalised parameter is probably negative")
            kl_sum += pk_norm[i] * np.log(math.fabs(pk_norm[i])/math.fabs((qk_norm[i] + self.tiny)) + self.tiny)

        return kl_sum


    @staticmethod
    def normalize(raw):
        norm = [float(i) / sum(raw) for i in raw]
        return norm
