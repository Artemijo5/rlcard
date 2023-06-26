import numpy as np
import collections

import os
import pickle

from rlcard.utils.utils import *

class PolicyIterator():
    '''Implement Policy Iteration, for the simplified limit-holdem
    '''

    def __init__(self, init_pi = None, gamma = 1.0, epsilon = 1e-10, model_path='.pol_iter_model'):
        self.use_raw = False
        self.env = rlcard.make('limit-holdem')
        self.model_path = model_path

        # state encoding:
        self.possible_states = 20
        self.A_s = 0
        self.K_s = 1
        self.Q_s = 2
        self.J_s = 3
        self.T_s = 4
        self.first_round = 0
        self.high_card = 1
        self.double = 2
        self.triple = 3
        # state = 4*card + quantity/round

        self.policy = init_pi
        # If no policy is provided, assume a random policy with uniform probability
        if init_pi == None:
            self.policy = np.array([[1.0/self.env.num_actions]*self.env.num_actions for _ in range(self.possible_states)])
        
        # State-Action Average Reward table, which will be filled prior to the actual algorithm
        self.T_P = 100 # number of games to fill in P - if viable, make 100k
        # 5 cards, can be either in 1st round, as high card, double, or third -> 20 states
        # 4 possible (attempted) actions
        # In total: 80 state-action combos
        # 2 numerical data stored for each state-action: times encountered (n), amassed reward (r)
        self.Pn = np.zeros((20, self.env.num_actions), dtype = np.int8) # times s-a has been encountered
        self.Pr = np.zeros((20, self.env.num_actions), dtype = np.float64) # reward amassed from enounters of s-a

    
    def fillInTable(pi = self.policy, Pn = self.Pn, Pr = self.Pr, runs = self.T_P):
        for t in range(runs):
            '''
            Run 1 game.
            Get states & rewards.
            Fill in the appropriate slots of P.
            '''
            print('WIP')
    
    def runToFill(pi = self.policy):
        '''Run a single game, decide action according to policy.'''
        print('WIP')

    def decideAccordingToPolicy(pi = self.policy):
        '''Use pi to make a decision, if not legal then make appropriate decision'''
        print('WIP')


    def policyEval(pi = self.policy, Pn = self.Pn, Pr = self.Pr, gamma = 1.0, epsilon = 1e-10):
        '''Policy Evaluation'''
        print('WIP')
    
    def policyImprovement(V, Pn = self.Pn, Pr = self.Pr):
        '''Policy Improvement'''
        print('WIP')
    
    def policyIteration(Pn = self.Pn, Pr = self.Pr, gamma = 1.0, epsilon = 1e-10):
        '''Policy Iteration'''
        print('WIP')
    
    def valueIteration(Pn = self.Pn, Pr = self.Pr, gamma = 1.0, epsilon = 1e-10):
        '''Value Iteration'''
        print('WIP')


    def get_state(self, player_id):
        state = self.env.get_state(player_id)
        
    

    def step(self, state):
        print('WIP')

    def eval_step(self, state):
        print('WIP')
    
    def save(self):
        print('WIP')
    
    def load(self):
        print('WIP')