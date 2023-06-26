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
        self.POSSIBLE_STATES = 20
        self.A_s = 0
        self.K_s = 1
        self.Q_s = 2
        self.J_s = 3
        self.T_s = 4
        self.FIRST_ROUND = 0
        self.HIGH_CARD = 1
        self.DOUBLE = 2
        self.TRIPLE = 3
        # state = 4*card + quantity/round
        # technically 18 possible states, since J, T can't be high cards, but would be tedious to change now

        self.policy = init_pi
        # If no policy is provided, assume a random policy with uniform probability
        if init_pi == None:
            self.policy = np.array([[1.0/self.env.num_actions]*self.env.num_actions for _ in range(self.POSSIBLE_STATES)])
        
        # State-Action Average Reward table, which will be filled prior to the actual algorithm
        self.T_P = 100 # number of games to fill in P - if viable, make 100k
        # 5 cards, can be either in 1st round, as high card, double, or third -> 20 states
        # 4 possible (attempted) actions
        # In total: 80 state-action combos
        # 2 numerical data stored for each state-action: times encountered (n), amassed reward (r)
        self.Pn = np.zeros((20, self.env.num_actions), dtype = np.int8) # times s-a has been encountered
        self.Pr = np.zeros((20, self.env.num_actions), dtype = np.float64) # reward amassed from enounters of s-a

        # Given the current state, probability of each other state being next (action shouldn't influence this)
        #self.P_next = np.zeros((self.POSSIBLE_STATES, self.POSSIBLE_STATES), dtype = np.float64)
        self.P_next = {
            0: {
                0 : 0, # First A
                1 : 0.48, # High A
                2 : 0.32, # Double A
                3 : 0.04, # Triple A
                4 : 0, # First K
                5 : 0, # High K
                6 : 0.04, # Double K
                7 : 0, # Triple K
                8 : 0, # First Q
                9 : 0, # High Q
                10: 0.04, # Double Q
                11: 0, # Triple Q
                12: 0, # First J
                13: 0, # High J
                14: 0.04, # Double J
                15: 0, # Triple J
                16: 0, # First T
                17: 0, # High T
                18: 0.04, # Double T
                19: 0 #Triple T
            }
        }

    
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
        '''Get the "state" of the player, ie the highest card in their hand + quantity/round'''
        state = self.env.get_state(player_id)
        state = state['raw_obs']
        hand = state['hand'][1]
        table = [elem[1] for elem in state['public_cards']]
        quant = 0
        card = 0

        card_states = {'A': self.A_s, 'K': self.K_s, 'Q': self.Q_s, 'J': self.J_s, 'T': self.T_s}

        if(len(total_state['public_cards'])==0):
            quant = self.FIRST_ROUND
            card = card_states(hand)
        else:
            if(table[0]==hand or table[1]==hand):
                card = card_states(hand)
                if table[0] == table[1]:
                    quant = self.TRIPLE
                else:
                    quant = self.DOUBLE
            elif(table[0] == table[1]):
                card = card_states(table[0])
                quant = self.DOUBLE
            else: # if all 3 cards are different, J and T are excluded, only A, K, Q matter
                quant = self.HIGH_CARD
                if(hand == 'A' or table[0] == 'A' or table[1] == 'A'):
                    card = card_states('A')
                elif(hand == 'K' or table[0] == 'K' or table[1] == 'K'):
                    card = card_states('K')
                else:
                    card = card_states('Q')
        
        return 4*card + quant

    def get_reward(self, player_id):
        reward = self.env.get_payoffs()[player_id]
        return reward
        
        
    

    def step(self, state):
        print('WIP')

    def eval_step(self, state):
        print('WIP')
    
    def save(self):
        print('WIP')
    
    def load(self):
        print('WIP')