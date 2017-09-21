import theano
import numpy as np
import theano.tensor as T
from NEATAgent.NEATEMAgent import NeatEMAgent

if __name__ == '__main__':
    agent = NeatEMAgent(40,2,4,40)

    # Test number of actions
    num_actions = 2
    # Test dimension
    dimension = 40



    # Our policy parameter is represented as a matrix where each column represents one action
    policy_parameter = np.random.rand(dimension, num_actions)
    value_parameter = np.ones(dimension)

    # define theano symbolic variables
    theta = theano.shared(policy_parameter, 'theta')
    omega = theano.shared(value_parameter, 'omega')
    phi = T.dmatrix('phi')
    action = T.imatrix('action')
    phi_new = T.dmatrix('phi_new')
    reward = T.dvector('reward')

    # Create symbolic function(s)
    # logpi(a|s)
    logpi = T.log(T.batched_dot(T.nnet.softmax(T.dot(phi, theta)), action))

    """
    Example of using logpi:
    >>> a = theano.function([phi, action], logpi)
    >>> a([[1.0, 1.0, 2.0], [1.0, 1.0, 2.0]], [[1, 0], [0, 1]])
    array([-1.08169047, -0.41403462])
    """

    # multiply logpi with td_error
    td_error = reward + 0.99 * T.dot(phi_new, omega) - T.dot(phi, omega)
    l_td_error = logpi * td_error
    l_td_error_mean = T.mean(l_td_error)
    logpi_func = theano.function([phi, action], logpi)
    logpi_td_error_func = theano.function([phi, phi_new, action, reward], l_td_error)
    logpi_td_error_mean_func = theano.function([phi, phi_new, action, reward], l_td_error_mean)
    td_error_func = theano.function([phi, phi_new, reward], td_error)
    # then do derivation to get e
    e = T.grad(l_td_error_mean, theta)
    e_func = theano.function([phi, phi_new, action, reward], e)

    e_squared = T.dot(T.transpose(e), e)
    e_squared_func = theano.function([phi, phi_new, action, reward], e_squared)

    e_sqr = T.sum(T.sqr(e).flatten(), axis=0)
    e_sqr_func = theano.function([phi, phi_new, action, reward], e_sqr)

    de_squared = T.grad(T.sum(T.sqr(e).flatten(), axis=0), theta)

    # d = theano.function([phi, phi_new, reward, action], de_squared)

    delta_policy = theano.function([phi, phi_new, action, reward], de_squared)


    # de_squared = T.grad(T.dot(e, e), theta)

    # de_squared = T.sum(T.jacobian(T.sqr(e).flatten(), theta), axis=0)

    # d = theano.function([phi, phi_new, reward, action], de_squared)

    # delta_policy = theano.function([phi, phi_new, action, reward], de_squared)

    # Example of using delta_policy function
    test_phi = np.ones(dimension*num_actions)
    test_phi_new = np.zeros(dimension*num_actions)
    for i in range(int(len(test_phi_new) / num_actions)):
        test_phi_new[i] = 1.0
    #     test_phi_new[i] = np.random.uniform(0, 1)

    test_phi = np.split(test_phi, num_actions)
    test_phi_new = np.split(test_phi_new, num_actions)
    test_action = [[1, 0], [0, 1]]  # each row contains ony 1 action
    test_reward = [1, 1]
    print("logpi")
    print(logpi_func(test_phi, test_action))
    print("td_error")
    print(td_error_func(test_phi, test_phi_new, test_reward))

    print("logpi * td_error")
    print(logpi_td_error_func(test_phi, test_phi_new, test_action, test_reward))
    print("Mean of logpi * td_error")
    print(logpi_td_error_mean_func(test_phi, test_phi_new, test_action, test_reward))
    print("e")
    print(e_func(test_phi, test_phi_new, test_action, test_reward))
    print("eT dot e")
    print(e_squared_func(test_phi, test_phi_new, test_action, test_reward))
    print("e squared")
    print(e_sqr_func(test_phi, test_phi_new, test_action, test_reward))
    print("delta policy")
    print(delta_policy(test_phi, test_phi_new, test_action, test_reward))


    #
    # dlogpi = T.jacobian(logpi, theta)
    #
    #
    # """
    # Example of using dlogpi:
    # >>> b = theano.function([phi, action], dlogpi)
    # >>> b([[1.0, 1.0, 2.0], [1.0, 1.0, 2.0]], [[1, 0], [0, 1]])
    # array([[[ 0.66097807, -0.66097807],
    #         [ 0.66097807, -0.66097807],
    #         [ 1.32195613, -1.32195613]],
    #        [[-0.33902193,  0.33902193],
    #         [-0.33902193,  0.33902193],
    #         [-0.67804387,  0.67804387]]])
    #
    # # here we have dlogpi for two sets of phi and action pairs
    # """
    # td_error = reward + T.dot(phi_new, omega) - T.dot(phi, omega)
    #
    # """
    # Example of using td_error:
    # >>> c = theano.function([phi, phi_new, reward], td_error)
    # >>> c([[1.0, 1.0, 2.0], [1.0, 1.0, 2.0]], [[1.0, 2.0, 2.0], [1.0, 2.0, 2.0]], [1.0, 1.0])
    # array([ 2.,  2.])
    # """
    # e = T.mean(T.batched_dot(dlogpi, td_error), axis=0)
    #
    # """
    # Example of using e:
    # >>> d = theano.function([phi, phi_new, reward, action], e)
    # >>> d([[1.0, 1.0, 2.0], [1.0, 1.0, 2.0]], [[1.0, 2.0, 2.0], [1.0, 2.0, 2.0]], [1, 1],[[1, 0], [0, 1]])
    # array([[ 0.32195613, -0.32195613],
    #    [ 0.32195613, -0.32195613],
    #    [ 0.64391226, -0.64391226]])
    #
    # # HERE E is the error value. Essentially it is the error policy parameter.
    # # THIS VALUE IS USED IN LITERATURE. BUT I will be trying to minimise the error function itself.
    # """
    #
    # # TODO ask AARON: THIS IS WHERE I AM CONFUSED.
    # # I am trying to differentiate error squared w.r.t theta, but I need to flatten error squared so that I can calculate the derivative
    # de_squared = T.grad(T.sum(T.sqr(e)), theta)
    #
    # delta_policy = theano.function([phi, phi_new, action, reward], de_squared)
    #
    # # Example of using delta_policy function
    # test_phi = [[1.0, 1.0, 2.0], [1.0, 1.0, 2.0]]
    # test_phi_new = [[1.0, 2.0, 2.0], [1.0, 2.0, 2.0]]
    # test_action = [[1, 0], [0, 1]]  # each row contains ony 1 action
    # test_reward = [1, 1]
    # print(delta_policy(test_phi, test_phi_new, test_action, test_reward))
    # '''
    # Example of result from delta_policy:
    # >>>
    #     [[[ 0.02912492 -0.02912492]
    #       [ 0.02912492 -0.02912492]
    #       [ 0.05824984 -0.05824984]]
    #
    #      [[ 0.02912492 -0.02912492]
    #       [ 0.02912492 -0.02912492]
    #       [ 0.05824984 -0.05824984]]
    #
    #      [[ 0.02912492 -0.02912492]
    #       [ 0.02912492 -0.02912492]
    #       [ 0.05824984 -0.05824984]]
    #
    #      [[ 0.02912492 -0.02912492]
    #       [ 0.02912492 -0.02912492]
    #       [ 0.05824984 -0.05824984]]
    #
    #      [[ 0.11649969 -0.11649969]
    #       [ 0.11649969 -0.11649969]
    #       [ 0.23299937 -0.23299937]]
    #
    #      [[ 0.11649969 -0.11649969]
    #       [ 0.11649969 -0.11649969]
    #       [ 0.23299937 -0.23299937]]]
    # '''
