from sklearn.neural_network import MLPClassifier
import sklearn
import sknn
from sklearn import datasets, linear_model
from sklearn import neural_network

import numpy as np
if __name__ == '__main__':
    dimension = 10
    num_actions = 2

    state_example = np.random.rand(4)
    state = [np.random.rand(4)for i in range(2)]

    phi_example = np.random.rand(dimension)
    test_phi = [np.random.rand(dimension) for i in range(2)]


    clf = neural_network.MLPRegressor(alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(state, test_phi)

    test_phi = np.ones(dimension * num_actions)
    test_phi_new = np.zeros(dimension * num_actions)
    for i in range(int(len(test_phi_new) / num_actions)):
        test_phi_new[i] = 1.0

    test_phi = np.split(test_phi, num_actions)
    test = np.random.rand(4)
    test2 = np.random.rand(4)
    print(test)
    print(clf.predict([test]).flatten())
    print(clf.predict([test2])[0])



