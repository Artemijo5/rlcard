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

        self.evaluated = False

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
            self.policy = np.array([[[[1.0/self.env.num_actions]*self.env.num_actions]*self.POSSIBLE_STATES]*4 for _ in range(2)])
        
        # State-Action Average Reward table, which will be filled prior to the actual algorithm
        # Assumes random adversary
        # 5 cards, can be either in 1st round, as high card, double, or third -> 20 states
        # 4 possible (attempted) actions
        # In total: 80 state-action combos
        self.R = np.zeros((2, 4,self.POSSIBLE_STATES, self.env.num_actions), dtype = np.float64) # reward amassed from enounters of s-a
        self.P1 = 0
        self.P2 = 1
        # self.FIRST_ROUND also works for the index here
        self.NONE_RAISED = 1
        self.ONE_RAISED = 2
        self.TWO_RAISED = 3

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

    def decideAccordingToPolicy(s, pi = self.policy):
        '''Use pi to make a decision, if not legal then make appropriate decision'''
        print('WIP')



    def policyEval(player_id = self.P1, pi = self.policy, P = self.P, R = self.R, gamma = 1.0, epsilon = 1e-10):
        '''
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
        '''
        print('WIP')

    def policyImprovement(V, player_id = 0, Pn = self.Pn, Pr = self.Pr):
        '''
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
        '''
        print('WIP')
    
    def policyIteration(player_id = 0, P = self.P, R = self.R, gamma = 1.0, epsilon = 1e-10):
        '''
        t = 0

        while True:
            old_pi = {s: pi(s) for s in range(len(P))}  #keep the old policy to compare with new
            self.fillInRewardTableRandom(old_pi, R)
            V = policy_evaluation(player_id, pi, P, R, gamma, epsilon)   #evaluate latest policy --> you receive its converged value function
            pi = policy_improvement(player_id, V, P, R, gamma)          #get a better policy using the value function of the previous one just calculated 
            
            t += 1
            Pplot[:,t]= [pi(s) for s in range(len(P))]  #keep track of the policy evolution
            Vplot[:,t] = V                              #and the value function evolution (for the GUI)
        
            if old_pi == {s:pi(s) for s in range(len(P))}: # you have converged to the optimal policy if the "improved" policy is exactly the same as in the previous step
                break
        print('converged after %d iterations' %t) #keep track of the number of (outer) iterations to converge
        self.evaluated = True
        return V,pi
        '''
        print('WIP')
    
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

    def get_state_Sim(hand, table):
        cards = {'A': self.A_s, 'K': self.K_s, 'Q': self.Q_s, 'J': self.J_s, 'T': self.T_s}
        quant = {'F': self.FIRST_ROUND, 'H': self.HIGH_CARD, 'D': self.DOUBLE, 'T': self.TRIPLE}

        if(table == None or len(table)==0):
            return 4*cards[hand] + quant['F']
        else:
            if(table[0]==hand or table[1]==hand):
                if(table[0]==table[1]):
                    return 4*cards[hand] + quant['T']
                else:
                    return 4*cards[hand] + quant['D']
            elif(table[0]==table[1]):
                return 4*cards[table[0]] + quant['D']
            else:
                if(hand == 'A' or table[0] == 'A' or table[1] == 'A'):
                    return 4*cards['A'] + quant['H']
                elif(hand == 'K' or table[0] == 'K' or table[1] == 'K'):
                    return 4*cards['K'] + quant['H']
                else:
                    return 4*cards['Q'] + quant['H']
    
    def compareHands(hand1, hand2, table):
        '''Return 1 if hand 1 is higher, -1 if hand 2 is higher, 0 if hands are equal'''
        state1 = get_state_Sim(hand1, table)
        state2 = get_state_Sim(hand2, table)
        cards = {'A': self.A_s, 'K': self.K_s, 'Q': self.Q_s, 'J': self.J_s, 'T': self.T_s}

        if state1 == state2:
            if hand1 == hand2:
                return 0
            elif cards[hand1] > cards[hand2]:
                return -1
            else:
                return 1
        elif state1 %4 == state2 %4:
            if state1 > state2:
                return -1
            else:
                return 1
        elif state1 %4 > state2 %4:
            return 1
        else:
            return -1


    def get_reward(self, player_id):
        reward = self.env.get_payoffs()[player_id]
        return reward
        

    def fillInRewardTableRandom(init_policy = self.policy, R = self.R):
            action = {'call': 0, 'raise': 1, 'fold': 2, 'check': 3}
            
            '''
            Exhaustively simulate every possible game state with current policy, to fill in table of rewards.
            '''
            # For player 1, round 1:
            Pn1 = np.zeros((20, 4), dtype = np.int8) # times s-a has been encountered
            Pr1 = np.zeros((20, 4), dtype = np.float64) # reward amassed from enounters of s-a
            # For player 2, round 1:
            Pn2 = np.zeros((20, 4), dtype = np.int8) # times s-a has been encountered
            Pr2 = np.zeros((20, 4), dtype = np.float64) # reward amassed from enounters of s-a
            # For both players, round 2, tracking how much has been raised:
            Pn = np.zeros((2, 3, 20, 4), dtype = np.int8) # times s-a has been encountered
            Pr = np.zeros((2, 3, 20, 4), dtype = np.float64) # average reward from enounters of s-a

            # First, 2 loops to cover all the first round scenarios:
            tokens_in = 0.5 # ante
            # wrt first round, only fold really matters

            # first, agent as player 1
            for hand1 in {'A', 'K', 'Q', 'J', 'T'}: # agent
                state = get_state_Sim(hand1, None)
                chanceOfRaisingFirst = init_policy[0][0][state][action['raise']] / (1 - init_policy[0][0][state][action['call']])
                chanceOfFoldingAfter = init_policy[0][0][state][action['fold']] / (init_policy[0][0][state][action['call']] + init_policy[0][0][state][action['fold']])
                chanceOFCallingAfter = init_policy[0][0][state][action['call']] / (init_policy[0][0][state][action['call']] + init_policy[0][0][state][action['fold']])
                for hand2  in {'A', 'K', 'Q', 'J', 'T'}: #adversary
                    for action1 in {'call', 'raise', 'fold', 'check'}: # agent
                        for action2 in {'call', 'raise', 'fold', 'check'}: # adversary
                            if(action1 == 'call' and action2 == 'raise'): # action1 can only be call after 2 was raise
                                Pn1[state][action[action1]] = 1 + Pn1[state][action[action1]]
                            # no immediate reward or loss
                            elif(action1 == 'raise'):
                                # if player 2 folds, immediate reward
                                # if player 2 raises, chance of folding afterwards -> loss of raised tokens
                                # if player 2 calls, no immediate reward or loss
                                if(action2 == 'fold'):
                                    Pn1[state][action[action1]] = 1 + Pn1[state][action[action1]]
                                    Pr1[state][action[action1]] += tokens_in
                                elif(action2 == 'raise'):
                                    Pn1[state][action[action1]] = 1 + Pn1[state][action[action1]]
                                    Pr1[state][action[action1]] -= (tokens_in + 1)*chanceOfFoldingAfter
                            elif(action1 == 'fold'):
                                if(action2 == 'raise'):
                                    # there is a chance we fold before raise, or after
                                    # if after, there is a chance we had raised or checked earlier
                                    Pn1[state][action[action1]] = 1 + Pn1[state][action[action1]]
                                    Pr1[state][action[action1]] -= tokens_in + chanceOfRaisingFirst*1
                                else:
                                    # loss is tokens_in
                                    Pn1[state][action[action1]] = 1 + Pn1[state][action[action1]]
                                    Pr1[state][action[action1]] -= tokens_in
                            else: # action1 == 'check'
                                if(action2 == 'raise'):
                                    # there is a chance we fold afterwards and lose the initial token
                                    Pn1[state][action[action1]] = 1 + Pn1[state][action[action1]]
                                    Pr1[state][action[action1]] -= (tokens_in)*chanceOfFoldingAfter
                                elif(action2 == 'fold'): # immediate reward of the initial token
                                    Pn1[state][action[action1]] = 1 + Pn1[state][action[action1]]
                                    Pr1[state][action[action1]] += tokens_in
                                elif(action2 == 'check'): # cannot be call
                                    Pn1[state][action[action1]] = 1 + Pn1[state][action[action1]]

            # secondly, agent as player 2
            for hand1 in {'A', 'K', 'Q', 'J', 'T'}: # adversary
                for hand2  in {'A', 'K', 'Q', 'J', 'T'}: # agent
                    state = get_state_Sim(hand2, None)
                    for action1 in {'call', 'raise', 'fold', 'check'}: # adversary
                        for action2 in {'call', 'raise', 'fold', 'check'}: # agent
                            # ignore case where p1 calls - no reward or loss from there
                            if action1 == 'raise':
                                if action2 == 'call':
                                    # no immediate reward or loss
                                    Pn2[state][action[action2]] = 1 + Pn2[state][action[action2]]
                                elif action2 == 'raise':
                                    # 50% chance that p1 folds afterwards
                                    Pn2[state][action[action2]] = 1 + Pn2[state][action[action2]]
                                    Pr2[state][action[action2]] += 0.5*(tokens_in + 1)
                                elif action2 == 'fold':
                                    Pn2[state][action[action2]] = 1 + Pn2[state][action[action2]]
                                    Pr2[state][action[action2]] -= tokens_in
                            elif action1 == 'fold':
                                if action2 == 'raise':
                                    # 50% chance the fold is done after the raise
                                    # in which case, 50% chance there was already a raise before
                                    # the other 50% chance is that the fold is done before the raise
                                    Pn2[state][action[action2]] = 1 + Pn2[state][action[action2]]
                                    Pr2[state][action[action2]] += tokens_in + 0.25
                                else:
                                    Pn2[state][action[action2]] = 1 + Pn2[state][action[action2]]
                            elif action1 == 'check':
                                if action2 == 'check':
                                    # no immediate reward or loss
                                    Pn2[state][action[action2]] = 1 + Pn2[state][action[action2]]
                                elif action2 == 'raise':
                                    # 50% chance p1 will fold afterwards
                                    Pn2[state][action[action2]] = 1 + Pn2[state][action[action2]]
                                    Pr2[state][action[action2]] += 0.5*(tokens_in)
                                elif action2 == 'fold':
                                    Pn2[state][action[action2]] = 1 + Pn2[state][action[action2]]
                                    Pr2[state][action[action2]] -= tokens_in

            # now for round 2

            # first, if agent is player1
            for tokens_in in {0.5, 1.5, 2.5}: # all possibilities for initial tokens
                t_index = int(np.floor(tokens_in)) # representative of the number of raises in the first round
                for hand1 in {'A', 'K', 'Q', 'J', 'T'}: # agent
                    for table1 in {'A', 'K', 'Q', 'J', 'T'}:
                        for table2  in {'A', 'K', 'Q', 'J', 'T'}:
                            table = [table1, table2]
                            state = get_state_Sim(hand1, table)
                            for hand2  in {'A', 'K', 'Q', 'J', 'T'}: # adversary
                                win = compareHands(hand1, hand2, table) # 1 if p1 has better hand, -1 if p2 has better hand, 0 if same hand
                                for action1 in {'call', 'raise', 'fold', 'check'}: # agent
                                    for action2 in {'call', 'raise', 'fold', 'check'}: # adversary
                                        chanceOfRaisingFirst = init_policy[0][0][t_index + 1][action['raise']] / (1 - init_policy[0][0][state][action['call']])
                                        chanceOfFoldingAfter = init_policy[0][t_index + 1][state][action['fold']] / (init_policy[0][0][state][action['call']] + init_policy[0][0][state][action['fold']])
                                        chanceOfCallingAfter = init_policy[0][t_index + 1][state][action['call']] / (init_policy[0][0][state][action['call']] + init_policy[0][0][state][action['fold']])
                                        if action1 == 'call':
                                            if action2 == 'raise':
                                                # p1 can call only after p2 raised
                                                # there is a chance p1 had raised before
                                                # win or lose 1 raised token, + a second raised token times that chance
                                                Pn[0][t_index][state][action[action1]] = 1 + Pn[0][t_index][state][action[action1]]
                                                Pr[0][t_index][state][action[action1]] += (tokens_in + 1)*win*chanceOfRaisingFirst
                                        elif action1 == 'raise':
                                            if(action2 == 'raise'):
                                                # there is a chance we fold afterwards and lose the raised token
                                                # or we call, and win or lose the doubly raised token
                                                Pn[0][t_index][state][action[action1]] = 1 + Pn[0][t_index][state][action[action1]]
                                                Pr[0][t_index][state][action[action1]] -= (tokens_in + 1)*chanceOfFoldingAfter
                                                Pr[0][t_index][state][action[action1]] += (tokens_in + 2)*win*chanceOfCallingAfter
                                            elif(action2 == 'fold'): # immediate reward of the initial token
                                                Pn[0][t_index][state][action[action1]] = 1 + Pn[0][t_index][state][action[action1]]
                                                Pr[0][t_index][state][action[action1]] += tokens_in
                                            elif(action2 == 'call'): # cannot be check
                                                Pn[0][t_index][state][action[action1]] = 1 + Pn[0][t_index][state][action[action1]]
                                                Pr[0][t_index][state][action[action1]] += (tokens_in+1)*win
                                        elif action1 == 'fold':
                                            if action2 == 'raise':
                                                # chance we had raised before
                                                Pn[0][t_index][state][action[action1]] = 1 + Pn[0][t_index][state][action[action1]]
                                                Pr[0][t_index][state][action[action1]] -= tokens_in + chanceOfRaisingFirst*1
                                            else:
                                                Pn[0][t_index][state][action[action1]] = 1 + Pn[0][t_index][state][action[action1]]
                                                Pr[0][t_index][state][action[action1]] -= tokens_in
                                        elif action1 == 'check':
                                            if(action2 == 'raise'):
                                                # there is a chance we fold afterwards and lose the initial token
                                                # or we call, and win or lose the raised token
                                                Pn[0][t_index][state][action[action1]] = 1 + Pn[0][t_index][state][action[action1]]
                                                Pr[0][t_index][state][action[action1]] -= (tokens_in)*chanceOfFoldingAfter
                                                Pr[0][t_index][state][action[action1]] += (tokens_in + 1)*win*chanceOfCallingAfter
                                            elif(action2 == 'fold'): # immediate reward of the initial token
                                                Pn[0][t_index][state][action[action1]] = 1 + Pn[0][t_index][state][action[action1]]
                                                Pr[0][t_index][state][action[action1]] += tokens_in
                                            elif(action2 == 'check'): # cannot be call
                                                Pn[0][t_index][state][action[action1]] = 1 + Pn[0][t_index][state][action[action1]]
                                                Pr[0][t_index][state][action[action1]] += tokens_in*win

            # secondly, if agent is p2
            for tokens_in in {0.5, 1.5, 2.5}: # all possibilities for initial tokens
                t_index = int(np.floor(tokens_in)) # representative of the number of raises in the first round
                for hand1 in {'A', 'K', 'Q', 'J', 'T'}: # adversary
                    for table1 in {'A', 'K', 'Q', 'J', 'T'}:
                        for table2  in {'A', 'K', 'Q', 'J', 'T'}:
                            table = [table1, table2]
                            for hand2  in {'A', 'K', 'Q', 'J', 'T'}: # agent
                                state = get_state_Sim(hand2, table)
                                win = compareHands(hand2, hand1, table) # 1 if p2 has better hand, -1 if p1 has better hand, 0 if same hand
                                for action1 in {'call', 'raise', 'fold', 'check'}: # adversary
                                    for action2 in {'call', 'raise', 'fold', 'check'}: # agent
                                    if action1 == 'call':
                                        if action2 == 'raise':
                                            # there is a 50% chance p1 had checked or raised earlier - 0.5 raised token on avg
                                            Pn[1][t_index][state][action[action2]] = 1 + Pn[1][t_index][state][action[action2]]
                                            Pr[1][t_index][state][action[action2]] += (tokens_in + 0.5)*win
                                    elif action1 == 'raise':
                                        if(action2 == 'raise'):
                                            # there is a 50% chance p1 will later call, 50% chance they will fold
                                            # there is a 50% chance p1 had checked or raised earlier - 0.5 raised token on avg
                                            # if call, see above case
                                            # if fold, p2 immediately wins 0.5 raised token
                                            Pn[1][t_index][state][action[action2]] = 1 + Pn[1][t_index][state][action[action2]]
                                            Pr[1][t_index][state][action[action2]] += 0.5*(tokens_in + 0.5)*win
                                            Pr[1][t_index][state][action[action2]] += 0.5*(tokens_in)*win
                                        elif(action2 == 'fold'): # immediate loss of the initial token
                                            Pn[1][t_index][state][action[action2]] = 1 + Pn[1][t_index][state][action[action2]]
                                            Pr[1][t_index][state][action[action2]] -= tokens_in
                                        elif(action2 == 'call'): # cannot be check
                                            # 1 raised token
                                            Pn[1][t_index][state][action[action2]] = 1 + Pn[1][t_index][state][action[action2]]
                                            Pr[1][t_index][state][action[action2]] += (tokens_in + 1)*win
                                    elif action1 == 'fold':
                                        if action2 == 'raise':
                                            # 50% chance p1 had raised before
                                            Pn[1][t_index][state][action[action2]] = 1 + Pn[1][t_index][state][action[action2]]
                                            Pr[1][t_index][state][action[action2]] += tokens_in + 0.25
                                        else:
                                            Pn[1][t_index][state][action[action2]] = 1 + Pn[1][t_index][state][action[action2]]
                                            Pr[1][t_index][state][action[action2]] += tokens_in
                                    elif action1 == 'check':
                                        if(action2 == 'raise'):
                                            # 50% chance p1 will fold, 50% chance p1 will call
                                            Pn[1][t_index][state][action[action2]] = 1 + Pn[1][t_index][state][action[action2]]
                                            Pr[1][t_index][state][action[action2]] += 0.5*(tokens_in)
                                            Pr[1][t_index][state][action[action2]] += 0.5*(tokens_in+1)*win
                                        elif(action2 == 'fold'): # immediate loss of the initial token
                                            Pn[1][t_index][state][action[action2]] = 1 + Pn[1][t_index][state][action[action2]]
                                            Pr[1][t_index][state][action[action2]] -= tokens_in
                                        elif(action2 == 'check'): # cannot be call
                                            Pn[1][t_index][state][action[action2]] = 1 + Pn[1][t_index][state][action[action2]]
                                            Pr[1][t_index][state][action[action2]] += (tokens_in)*win
            


            for s in range(20):
                for a in range(4):
                    if Pn1[s][a] == 0:
                        R[self.P1][self.FIRST_ROUND][s][a] = 0.0
                    else:
                        R[self.P1][self.FIRST_ROUND][s][a] = Pr1[s][a] / np.abs(Pn1[s][a])

                    if Pn2[s][a] == 0:
                        R[self.P2][self.FIRST_ROUND][s][a] = 0.0
                    else:
                        R[self.P2][self.FIRST_ROUND][s][a] = Pr2[s][a] / np.abs(Pn2[s][a])

                    for pid in range(2):
                        for raised in range(3):
                            if Pn[pid][raised+1][s][a] == 0:
                                R[pid][raised+1][s][a] = 0.0
                            else:
                                R[pid][raised+1][s][a] = Pr[pid][raised][s][a] / np.abs(Pn[pid][raised][s][a])
  

    def step(self, state):
        print('WIP')

    def eval_step(self, state):
        print('WIP')
    
    def save(self):
        print('WIP')
    
    def load(self):
        print('WIP')