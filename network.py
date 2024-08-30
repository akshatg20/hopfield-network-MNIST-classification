import numpy as np
import math
from itertools import combinations, product


class myHopfieldNetwork(object):

    # constructor
    def __init__(self, nuerons):
        self.neurons = nuerons
        self.state =  ( 2 * np.random.randint(0, 2, self.neurons) - 1 )

    # function to learn weights from a pattern list
    def calculate_weights(self, pattern_list, rule = "hebb"):
        # condition to check whether all patterns in the list are compatible
        condition = all(len(p.flatten()) == self.neurons for p in pattern_list)

        if not condition:
            e = "Incompatible patterns"
            raise ValueError(e)
        
        # initialize weights
        weights = np.zeros((self.neurons,self.neurons))
        mean = np.sum([np.sum(i) for i in pattern_list]) / self.neurons

        if rule == "hebb":

            for p in pattern_list:
                p_flat = p.flatten()
                p_flat = p_flat - mean
                for i in range(self.neurons):
                    for j in range(self.neurons):
                        weights[i, j] += p_flat[i] * p_flat[j]
            weights /= self.neurons

            # self connections would be zero
            np.fill_diagonal(weights, 0)
        elif rule == "pseudo-inverse":
            patternN = np.array(pattern_list)
            pattern = patternN.reshape(patternN.shape[0], patternN.shape[1]*patternN.shape[2])
            c = np.tensordot( pattern, pattern, axes=( ( 1 ), ( 1 ) ) ) / len( pattern )
            cinv = np.linalg.inv( c )
            for k, l in product( range( len( pattern ) ), range( len( pattern ) ) ):
                weights = weights + cinv[ k, l ] * pattern[ k ] * pattern[ l ].reshape( ( -1, 1 ) )
            weights = weights / len( pattern )
        else:
            e = "Incompatible Rule"
            raise ValueError(e)

        return weights

    # function to compute evolution states
    def network_evolution(self, initial_state, weights, method = "sync", steps=5):
        self.state = initial_state
        states = list()
        states.append(self.state.copy())
        for i in range(steps):
            self.run(weights, method)
            states.append(self.state.copy())
        return states
    
    # function to compute a single evolution state
    def run(self, weights, method):
        if method == "async":
            self.state = _evolve_once_async(self.state, weights)
        else:
            self.state = _evolve_once_sync(self.state, weights)
    


# function to run evolution on a state once, implementing a synchronous state update using sign(h)
def _evolve_once_sync(state, weights):
    init_state = state.flatten()
    h = np.sum(init_state * weights, axis = 1)
    new_state = np.sign(h)
    zero_neurons = new_state == 0
    new_state[zero_neurons] = 1

    # modify the given state
    state = new_state.reshape(state.shape)
    return state


def _evolve_once_async(state, weights):
    init_state = state.flatten()
    random_neuron_idx_list = np.random.permutation(len(init_state))
    state_s1 = init_state.copy()
    for i in range(len(random_neuron_idx_list)):
        rand_neuron_i = random_neuron_idx_list[i]
        h_i = np.dot(weights[:, rand_neuron_i], state_s1)
        s_i = np.sign(h_i)
        if s_i == 0:
            s_i = 1
        state_s1[rand_neuron_i] = s_i
    
    state = state_s1.reshape(state.shape)
    return state