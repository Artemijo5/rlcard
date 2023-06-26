import numpy as np
import collections

import os
import pickle

from rlcard.utils.utils import *

class PolicyIterator():
    '''Implement Policy Iteration, for the simplified limit-holdem
    '''

    def __init__(self, env, init_pi = None, P, gamma = 1.0, epsilon = 1e-10, model_path='.pol_iter_model'):
        self.use_raw = False
        self.env = env
        self.model_path = model_path

        self.policy = init_pi
        if init_pi == None:
            self.policy = np.array([1.0/self.env.num_actions for _ in range(self.env.num_actions)])