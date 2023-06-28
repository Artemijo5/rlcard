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
        # Assumes random adversary
        # 5 cards, can be either in 1st round, as high card, double, or third -> 20 states
        # 4 possible (attempted) actions
        # In total: 80 state-action combos
        self.R = np.zeros((2, 4,self.POSSIBLE_STATES, self.env.num_actions), dtype = np.float64) # reward amassed from enounters of s-a
        # TODO either fill in P's manually, or add the function that fills them on init
        self.P1 = 0
        self.P2 = 1
        # self.FIRST_ROUND also works for the index here
        self.NONE_RAISED = 1
        self.ONE_RAISED = 2
        self.TWO_RAISED = 3
        #self.fillInRewardTable()

        # State Transition Probability Table
        # only needed for first round states - second round states are guaranteed final
        # gamma will need to be less than one, to account for the possibility of game ending at round 1
        self.P_next = {
            0: { # First A
                0 : 0.0, # First A
                1 : 0.48, # High A
                2 : 0.32, # Double A
                3 : 0.04, # Triple A
                4 : 0.0, # First K
                5 : 0.0, # High K
                6 : 0.04, # Double K
                7 : 0.0, # Triple K
                8 : 0.0, # First Q
                9 : 0.0, # High Q
                10: 0.04, # Double Q
                11: 0.0, # Triple Q
                12: 0.0, # First J
                13: 0.0, # High J
                14: 0.04, # Double J
                15: 0.0, # Triple J
                16: 0.0, # First T
                17: 0.0, # High T
                18: 0.04, # Double T
                19: 0.0 #Triple T
            },
            4: { # First K
                0 : 0.0, # First A
                1 : 0.24, # High A
                2 : 0.04, # Double A
                3 : 0.0, # Triple A
                4 : 0.0, # First K
                5 : 0.24, # High K
                6 : 0.32, # Double K
                7 : 0.04, # Triple K
                8 : 0.0, # First Q
                9 : 0.0, # High Q
                10: 0.04, # Double Q
                11: 0.0, # Triple Q
                12: 0.0, # First J
                13: 0.0, # High J
                14: 0.04, # Double J
                15: 0.0, # Triple J
                16: 0.0, # First T
                17: 0.0, # High T
                18: 0.04, # Double T
                19: 0.0 #Triple T
            },
            8: { # First Q
                0 : 0.0, # First A
                1 : 0.24, # High A
                2 : 0.04, # Double A
                3 : 0.0, # Triple A
                4 : 0.0, # First K
                5 : 0.16, # High K
                6 : 0.04, # Double K
                7 : 0.04, # Triple K
                8 : 0.0, # First Q
                9 : 0.0, # High Q
                10: 0.04, # Double Q
                11: 0.0, # Triple Q
                12: 0.0, # First J
                13: 0.0, # High J
                14: 0.04, # Double J
                15: 0.0, # Triple J
                16: 0.0, # First T
                17: 0.0, # High T
                18: 0.04, # Double T
                19: 0.0 #Triple T
            },
            12: { # First J
                0 : 0.0, # First A
                1 : 0.24, # High A
                2 : 0.32, # Double A
                3 : 0.0, # Triple A
                4 : 0.0, # First K
                5 : 0.16, # High K
                6 : 0.04, # Double K
                7 : 0.0, # Triple K
                8 : 0.0, # First Q
                9 : 0.08, # High Q
                10: 0.04, # Double Q
                11: 0.0, # Triple Q
                12: 0.0, # First J
                13: 0.0, # High J
                14: 0.32, # Double J
                15: 0.04, # Triple J
                16: 0.0, # First T
                17: 0.0, # High T
                18: 0.04, # Double T
                19: 0.0 #Triple T
            },
            16: { # First T
                0 : 0.0, # First A
                1 : 0.24, # High A
                2 : 0.04, # Double A
                3 : 0.0, # Triple A
                4 : 0.0, # First K
                5 : 0.16, # High K
                6 : 0.04, # Double K
                7 : 0.0, # Triple K
                8 : 0.0, # First Q
                9 : 0.08, # High Q
                10: 0.04, # Double Q
                11: 0.0, # Triple Q
                12: 0.0, # First J
                13: 0.0, # High J
                14: 0.04, # Double J
                15: 0.0, # Triple J
                16: 0.0, # First T
                17: 0.0, # High T
                18: 0.32, # Double T
                19: 0.04 #Triple T
            }
        }

        # PLOTTING
        size = self.POSSIBLE_STATES
        self.Tmax = 100000
        Vplot = np.zeros((size,Tmax)) #these keep track how the Value function evolves, to be used in the GUI
        Pplot = np.zeros((size,Tmax)) #these keep track how the Policy evolves, to be used in the GUI
        t = 0

    
    def fillInRewardTable(P = self.P, R = self.R):
        for t in range(runs):
            # TODO either add exhaustive code here or fill reward table manually
            '''
            Run 1 game.
            Get states & rewards.
            Fill in the appropriate slots of P.
            '''
            print('WIP')


    def decideAccordingToPolicy(s, pi = self.policy):
        '''Use pi to make a decision, if not legal then make appropriate decision'''
        print('WIP')



    def policyEval(player_id = self.P1, pi = self.policy, P = self.P, R = self.R, gamma = 1.0, epsilon = 1e-10):
        actions = {'call', 'raise', 'fold', 'check'}
        action_code = {'call': 0, 'raise': 1, 'fold': 2, 'check': 3}

        t = 0
        prev_V = np.zeros(self.POSSIBLE_STATES)
        while True:
            V = np.zeros(self.POSSIBLE_STATES)
            for s in range(self.POSSIBLE_STATES): # do for every state
                for a in actions: # do for every our agent could take action
                    if s% 4 == 0: # First Round
                        # Bellman Step to include current round's reward
                        V[s] += (pi[s][action_code[a])*(R[player_id][self.FIRST_ROUND][s][action_code[a]] + gamma*prev_V[s]) # TODO ???
                        for next_state in range(len(P[s])):
                            prob = P[s][next_state]
                            reward = 0
                            for a2 in actions:
                                reward += pi[next_state][action_code[a2]]*R[player_id][self.FIRST_ROUND][next_state][action_code[a2]]
                            # Bellman Step
                            V[s] += prob * (reward + gamma*prev_V[next_state]) # TODO see how the 'not done' is to be handled
                    else:
                        # Second Round
                        # There is no next step, only include this case with the Bellman step
                        for t_index in {1, 2, 3}:
                            V[s] += (pi[s][action_code[a])*(R[player_id][t_index][s][action_code[a]] + gamma*prev_V[s]) # TODO ???
            if np.max(np.abs(V - prev_V)) < epsilon:
                break
            prev_V = V.copy()
            t += 1
            Vplot[:,t] = prev_V  # accounting for GUI
        return V

    def policyImprovement(V, player_id = 0, Pn = self.Pn, Pr = self.Pr):
        actions = {'call', 'raise', 'fold', 'check'}
        action_code = {'call': 0, 'raise': 1, 'fold': 2, 'check': 3}

        Q = np.zeros(len(P), len(P[0]))
        for s in range(len(P)):
            for a in actions:
                if s%4 == 0:
                    for next_state in range(len(P[s])):
                        prob = P[s][next_state]
                        reward = 0
                        for a2 in actions:
                            reward += pi[next_state][action_code[a2]]*R[player_id][self.FIRST_ROUND][next_state][action_code[a2]]
                        # Bellman Step
                        Q[s][action_code[a]] += prob * (reward + gamma * V[next_state]) # TODO ???
                else:
                    for t_index in {1, 2, 3}:
                        Q[s][action_code[a]] += (pi[s][action_code[a])*(R[player_id][t_index][s][action_code[a]] + gamma*prev_V[s]) # TODO ???
        new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

        return new_pi
    
    def policyIteration(player_id = 0, P = self.P, R = self.R, gamma = 1.0, epsilon = 1e-10):
        t = 0

        while True:
            old_pi = {s: pi(s) for s in range(len(P))}  #keep the old policy to compare with new
            V = policy_evaluation(player_id, pi, P, R, gamma, epsilon)   #evaluate latest policy --> you receive its converged value function
            pi = policy_improvement(player_id, V, P, R, gamma)          #get a better policy using the value function of the previous one just calculated 
            
            t += 1
            Pplot[:,t]= [pi(s) for s in range(len(P))]  #keep track of the policy evolution
            Vplot[:,t] = V                              #and the value function evolution (for the GUI)
        
            if old_pi == {s:pi(s) for s in range(len(P))}: # you have converged to the optimal policy if the "improved" policy is exactly the same as in the previous step
                break
        print('converged after %d iterations' %t) #keep track of the number of (outer) iterations to converge
        return V,pi
    
    def valueIteration(Pn = self.Pn, Pr = self.Pr, gamma = 1.0, epsilon = 1e-10):
        '''Value Iteration'''
        print('WIP')


    def get_state(self, player_id):
        '''Get the "state" of the player, ie the highest card in their hand + quantity/round'''
        state = self.env.get_state(player_id)
        state = state['raw_obs']
        hand = state['hand'][0][1]
        table = [elem[1] for elem in state['public_cards']]
        quant = 0
        card = 0

        card_states = {'A': self.A_s, 'K': self.K_s, 'Q': self.Q_s, 'J': self.J_s, 'T': self.T_s}

        if(len(table)==0):
            quant = self.FIRST_ROUND
            card = card_states[hand]
        else:
            if(table[0]==hand or table[1]==hand):
                card = card_states[hand]
                if table[0] == table[1]:
                    quant = self.TRIPLE
                else:
                    quant = self.DOUBLE
            elif(table[0] == table[1]):
                card = card_states[table[0]]
                quant = self.DOUBLE
            else: # if all 3 cards are different, J and T are excluded, only A, K, Q matter
                quant = self.HIGH_CARD
                if(hand == 'A' or table[0] == 'A' or table[1] == 'A'):
                    card = card_states['A']
                elif(hand == 'K' or table[0] == 'K' or table[1] == 'K'):
                    card = card_states['K']
                else:
                    card = card_states['Q']
        
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