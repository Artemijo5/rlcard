import numpy as np

from rlcard.utils.utils import *

class PolicyIterator():
    '''Implement Policy Iteration, for the simplified limit-holdem
    '''

    def __init__(self, env = None, init_pi = None, gamma = 1.0, epsilon = 1e-10):
        self.use_raw = False
        self.env = env

        self.evaluated = [False, False] # for players 1 and 2
        self.num_actions = 4 # self.env.num_actions

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
            self.policy = np.array([[[[1.0/self.num_actions]*self.num_actions]*self.POSSIBLE_STATES]*4 for _ in range(2)])
        
        # State-Action Average Reward table, which will be filled prior to the actual algorithm
        # Assumes random adversary
        # 5 cards, can be either in 1st round, as high card, double, or third -> 20 states
        # 4 possible (attempted) actions
        # In total: 80 state-action combos
        self.R = np.zeros((2, 4,self.POSSIBLE_STATES, self.num_actions), dtype = np.float64) # reward amassed from enounters of s-a
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
        self.size = self.POSSIBLE_STATES
        self.Tmax = 100000
        self.Vplot = np.zeros((2, 4, self.size, self.Tmax)) #these keep track how the Value function evolves, to be used in the GUI
        self.Pplot = np.zeros((2, 4, self.size, 4, self.Tmax)) #these keep track how the Policy evolves, to be used in the GUI
        self.t = 0
    
    def policyEval(self, gamma = 1.0, epsilon = 1e-10):
        actions = {'call', 'raise', 'fold', 'check'}
        action_code = {'call': 0, 'raise': 1, 'fold': 2, 'check': 3}

        t = 0
        prev_V = np.zeros((2, 4, self.POSSIBLE_STATES)) # indexed per: player_id, raised, 'state'
        while True:
            V = np.zeros((2, 4, self.POSSIBLE_STATES))
            for player_id in [self.P1, self.P2]:
                for s in range(self.POSSIBLE_STATES): # do for every state
                    if s%4 == 0: # for first round
                        # THERE NEED BE 2 CASES:
                        # for P1
                        # and for P2

                        if player_id == self.P1: # player 1 case
                            for a1 in actions: # agent
                                ac = action_code[a1]
                                raised = 0
                                q = self.R[player_id][self.FIRST_ROUND][s][ac]
                                a_prob = self.policy[player_id][self.FIRST_ROUND][s][:] # keep track of policy

                                chance = a_prob[ac] # probability this decision was taken, according to policy
                                if(chance == 0.45 or chance == 0.55 or chance == 1): # after 1st iteration, policy is deterministic
                                    chance = 1
                                else:
                                    if a1 == 'call':
                                        chance = chance / (chance + a_prob[2])
                                    elif a1 == 'raise' or a1 == 'check':
                                        chance = chance / (a_prob[1] + a_prob[2] + a_prob[3])
                                    elif a1 == 'fold':
                                        chance = chance / (a_prob[1] + a_prob[2] + a_prob[3]) # prob it is before p2's turn
                                        chance += (1-chance)*(1/3)* a_prob[2]/(a_prob[0]+a_prob[2]) # + prob after p2 raises

                                for next_state in range(len(self.P_next[s])):
                                    prob = self.P_next[s]
                                    if(a1 != 'fold' and a2 != 'fold' and prob != 0 and chance != 0):
                                        for a2 in actions: # adversary
                                            if a1 == 'raise':
                                                raised += 1
                                            if a2 == 'raise':
                                                raised += 1
                                            
                                            if a1 == 'call':
                                                if a2 == 'raise':
                                                    q += chance*(1/3)*gamma*prob*prev_V[1 - player_id][raised+1][next_state]
                                                    # 1/3 factor represents adversary's policy
                                                    # 1-player_id because we are switching player positions in the next round
                                            
                                            if a1 == 'raise':
                                                if a2 == 'call':
                                                    q += chance*(1/3)*gamma*prob*prev_V[player_id][raised+1][next_state]
                                                elif a2 == 'raise':
                                                    q += chance*(1/3)*gamma*prob*prev_V[1-player_id][raised+1][next_state]
                                                # fold case is final and therefore ignored
                                            
                                            if a2 == 'check':
                                                if a2 == 'check':
                                                    q += chance*(1/3)*gamma*prob*prev_V[player_id][raised+1][next_state]
                                                elif a2 == 'raise':
                                                    q += chance*(1/3)*gamma*prob*prev_V[1-player_id][raised+1][next_state]
                                                # fold case is final and therefore ignored
                                            
                                            raised = 0 # at the end, reset raised

                                V[player_id][self.FIRST_ROUND][s] += chance*q

                        if player_id == self.P2:
                            for a2 in actions: # agent
                                ac = action_code[a2]
                                raised = 0
                                q = self.R[player_id][self.FIRST_ROUND][s][ac]
                                a_prob = self.policy[player_id][self.FIRST_ROUND][s][:] # keep track of policy

                                chance = a_prob[ac] # probability this decision was taken, according to policy
                                if(chance == 0.45 or chance == 0.55 or chance == 1): # after 1st iteration, policy is deterministic
                                    chance = 1
                                else:
                                    if a2 == 'call':
                                        chance = chance / (a_prob[0] + a_prob[1] + a_prob[2])
                                    elif a2 == 'raise' or a2 == 'fold':
                                        chance = chance / (0.5*a_prob[0] + a_prob[1] + a_prob[2] + 0.5*a_prob[3])
                                    elif a2 == 'check':
                                        chance = chance / (a_prob[1] + a_prob[2] + a_prob[3])

                                for a1 in actions: # adversary

                                    for next_state in range(len(self.P_next[s])):
                                        prob = self.P_next[s]
                                        if(a1 != 'fold' and a2 != 'fold' and prob != 0):
                                            if a1 == 'raise':
                                                raised += 1
                                            if a2 == 'raise':
                                                raised += 1
                                            
                                            if a1 == 'call':
                                                if a2 == 'raise':
                                                    q += chance*(1/2)*gamma*prob*prev_V[1-player_id][raised+1][next_state]
                                                
                                            if a1 == 'raise':
                                                if a2 == 'call':
                                                    q += chance*(1/3)*gamma*prob*prev_V[player_id][raised+1][next_state]
                                                if a2 == 'raise':
                                                    q += chance*(1/3)*gamma*prob*prev_V[1 - player_id][raised+1][next_state]

                                            if a1 == 'check':
                                                if a2 == 'call':
                                                    q += chance*(1/3)*gamma*prob*prev_V[player_id][raised+1][next_state]
                                                if a2 == 'raise':
                                                    q += chance*(1/3)*gamma*prob*prev_V[1 - player_id][raised+1][next_state]


                                            raised = 0 # at the end, reset raised
                                    
                                V[player_id][self.FIRST_ROUND][s] += chance*q

                    else: # for second round
                        for raised in {0, 1, 2}: # for any number of tokens raised from round 1
                            for a in actions:
                                q = self.R[player_id][raised+1][s][action_code[a]]
                                a_prob = self.policy[player_id][self.FIRST_ROUND][s][:] # keep track of policy

                                chance = a_prob[action_code[a]]
                                
                                if(chance == 0.45 or chance == 0.55 or chance == 1): # deterministic
                                    chance = 1
                                elif player_id == self.P1:
                                    if a == 'call':
                                        chance = chance / (chance + a_prob[2])
                                    elif a == 'raise' or a == 'check':
                                        chance = chance / (a_prob[1] + a_prob[2] + a_prob[3])
                                    elif a == 'fold':
                                        chance = chance / (a_prob[1] + a_prob[2] + a_prob[3]) # prob it is before p2's turn
                                        chance += (1-chance)*(1/3)* a_prob[2]/(a_prob[0]+a_prob[2]) # + prob after p2 raises
                                elif player_id == self.P2:
                                    if a == 'call':
                                        chance = chance / (a_prob[0] + a_prob[1] + a_prob[2])
                                    elif a == 'raise' or a == 'fold':
                                        chance = chance / (0.5*a_prob[0] + a_prob[1] + a_prob[2] + 0.5*a_prob[3])
                                    elif a == 'check':
                                        chance = chance / (a_prob[1] + a_prob[2] + a_prob[3])
                                # game lasts two rounds, so no next state
                                V[player_id][raised+1][s] += chance*q
            if np.max(np.abs(V - prev_V)) < epsilon:
                break
            prev_V = V.copy()
            t += 1
            Vplot[:][:][:][t] = prev_V
            for j in range(4):
                self.Vplot[player_id, j,:,t] = prev_V[:,j]  # accounting for GUI
        return V

    def policyImprovement(self, V, gamma = 1.0):
        actions = {'call', 'raise', 'fold', 'check'}
        action_code = {'call': 0, 'raise': 1, 'fold': 2, 'check': 3}

        Q = np.zeros((2, 4, self.POSSIBLE_STATES, 4))
        new_pi = np.zeros((4, self.POSSIBLE_STATES, 4))
        for player_id in [self.P1, self.P2]:
            for s in range(self.POSSIBLE_STATES):
                if s%4 == 0: # for first round
                    # THERE NEED BE 2 CASES:
                    # for P1
                    # and for P2

                    if player_id == self.P1: # player 1 case
                            for a1 in actions: # agent
                                ac = action_code[a1]
                                raised = 0
                                q = self.R[player_id][self.FIRST_ROUND][s][ac]
                                a_prob = self.policy[player_id][self.FIRST_ROUND][s][:] # keep track of policy

                                chance = a_prob[ac] # probability this decision was taken, according to policy
                                if(chance == 0.45 or chance == 0.55 or chance == 1): # after 1st iteration, policy is deterministic
                                    chance = 1
                                else:
                                    if a1 == 'call':
                                        chance = chance / (chance + a_prob[2])
                                    elif a1 == 'raise' or a1 == 'check':
                                        chance = chance / (a_prob[1] + a_prob[2] + a_prob[3])
                                    elif a1 == 'fold':
                                        chance = chance / (a_prob[1] + a_prob[2] + a_prob[3]) # prob it is before p2's turn
                                        chance += (1-chance)*(1/3)* a_prob[2]/(a_prob[0]+a_prob[2]) # + prob after p2 raises

                                for next_state in range(len(self.P_next[s])):
                                    prob = self.P_next[s]
                                    if(a1 != 'fold' and a2 != 'fold' and prob != 0 and chance != 0):
                                        for a2 in actions: # adversary
                                            if a1 == 'raise':
                                                raised += 1
                                            if a2 == 'raise':
                                                raised += 1
                                            
                                            if a1 == 'call':
                                                if a2 == 'raise':
                                                    q += chance*(1/3)*gamma*prob*V[1 - player_id][raised+1][next_state]
                                                    # 1/3 factor represents adversary's policy
                                                    # 1-player_id because we are switching player positions in the next round
                                            
                                            if a1 == 'raise':
                                                if a2 == 'call':
                                                    q += chance*(1/3)*gamma*prob*V[player_id][raised+1][next_state]
                                                elif a2 == 'raise':
                                                    q += chance*(1/3)*gamma*prob*V[1-player_id][raised+1][next_state]
                                                # fold case is final and therefore ignored
                                            
                                            if a2 == 'check':
                                                if a2 == 'check':
                                                    q += chance*(1/3)*gamma*prob*V[player_id][raised+1][next_state]
                                                elif a2 == 'raise':
                                                    q += chance*(1/3)*gamma*prob*V[1-player_id][raised+1][next_state]
                                                # fold case is final and therefore ignored
                                            
                                            raised = 0 # at the end, reset raised

                                Q[player_id][self.FIRST_ROUND][s][ac] += chance*q

                    if player_id == self.P2:
                            for a2 in actions: # agent
                                ac = action_code[a2]
                                raised = 0
                                q = self.R[player_id][self.FIRST_ROUND][s][ac]
                                a_prob = self.policy[player_id][self.FIRST_ROUND][s][:] # keep track of policy

                                chance = a_prob[ac] # probability this decision was taken, according to policy
                                if(chance == 0.45 or chance == 0.55 or chance == 1): # after 1st iteration, policy is deterministic
                                    chance = 1
                                else:
                                    if a2 == 'call':
                                        chance = chance / (a_prob[0] + a_prob[1] + a_prob[2])
                                    elif a2 == 'raise' or a2 == 'fold':
                                        chance = chance / (0.5*a_prob[0] + a_prob[1] + a_prob[2] + 0.5*a_prob[3])
                                    elif a2 == 'check':
                                        chance = chance / (a_prob[1] + a_prob[2] + a_prob[3])

                                for a1 in actions: # adversary

                                    for next_state in range(len(self.P_next[s])):
                                        prob = self.P_next[s]
                                        if(a1 != 'fold' and a2 != 'fold' and prob != 0):
                                            if a1 == 'raise':
                                                raised += 1
                                            if a2 == 'raise':
                                                raised += 1
                                            
                                            if a1 == 'call':
                                                if a2 == 'raise':
                                                    q += chance*(1/2)*gamma*prob*V[1-player_id][raised+1][next_state]
                                                
                                            if a1 == 'raise':
                                                if a2 == 'call':
                                                    q += chance*(1/3)*gamma*prob*V[player_id][raised+1][next_state]
                                                if a2 == 'raise':
                                                    q += chance*(1/3)*gamma*prob*V[1 - player_id][raised+1][next_state]

                                            if a1 == 'check':
                                                if a2 == 'call':
                                                    q += chance*(1/3)*gamma*prob*V[player_id][raised+1][next_state]
                                                if a2 == 'raise':
                                                    q += chance*(1/3)*gamma*prob*V[1 - player_id][raised+1][next_state]


                                            raised = 0 # at the end, reset raised
                                    
                                Q[player_id][self.FIRST_ROUND][s][ac] += chance*q
                   
                else: # for second round
                    for raised in {0, 1, 2}: # for any number of tokens raised from round 1
                            for a in actions:
                                q = self.R[player_id][raised+1][s][action_code[a]]
                                ac = action_code[a]
                                a_prob = self.policy[player_id][self.FIRST_ROUND][s][:] # keep track of policy

                                chance = a_prob[ac]
                                
                                if(chance == 0.45 or chance == 0.55 or chance == 1): # deterministic
                                    chance = 1
                                elif player_id == self.P1:
                                    if a == 'call':
                                        chance = chance / (chance + a_prob[2])
                                    elif a == 'raise' or a == 'check':
                                        chance = chance / (a_prob[1] + a_prob[2] + a_prob[3])
                                    elif a == 'fold':
                                        chance = chance / (a_prob[1] + a_prob[2] + a_prob[3]) # prob it is before p2's turn
                                        chance += (1-chance)*(1/3)* a_prob[2]/(a_prob[0]+a_prob[2]) # + prob after p2 raises
                                elif player_id == self.P2:
                                    if a == 'call':
                                        chance = chance / (a_prob[0] + a_prob[1] + a_prob[2])
                                    elif a == 'raise' or a == 'fold':
                                        chance = chance / (0.5*a_prob[0] + a_prob[1] + a_prob[2] + 0.5*a_prob[3])
                                    elif a == 'check':
                                        chance = chance / (a_prob[1] + a_prob[2] + a_prob[3])
                                # game lasts two rounds, so no next state
                                Q[player_id][raised+1][s][ac] += chance*q
        # TODO figure out how the following is supposed to work...
        new_pi = Q.copy()
        # to make deterministic, when choosing an action during play, we pick the one with the highest policy prob
        for raised in range(4):
            for s in range(self.POSSIBLE_STATES):
                pol = new_pi[raised][s]
                if(not np.sum(pol[:]) == 0):
                    # first, player1 case
                    action1 = 0
                    action2 = 0
                    if player_id == self.P1:
                        action1 = np.max([pol[1], pol[2], pol[3]])
                        action2 = np.max([pol[0], pol[2]])
                    else:
                        action1 = np.max([pol[1], pol[2], pol[3]])
                        action2 = np.max([pol[0], pol[1], pol[2]])
                    for a in range(4):
                        old = pol[a]
                        pol[a] = 0
                        if(old == action1):
                            pol[a] += 0.55
                        if(old == action2):
                            pol[a] += 0.45
                new_pi[raised][s] = pol[:]
        return new_pi
    # TODO
    # the problem lies here
    # first fix logic above to make sure it works correctly
    # but also, ensure each [s] of new_pi has 2 positions at most filled in
    # perhaps could be done by picking the action with highest immediate reward in case of multiplicity...
    
    def policyIteration(self, gamma = 1.0, epsilon = 1e-10):
        t = 0

        while True:
            old_pi = self.policy[player_id].copy()
            
            self.fillInRewardTableRandom()
            V = self.policyEval(gamma, epsilon)   #evaluate latest policy --> you receive its converged value function
            self.policy[player_id] = self.policyImprovement(V, gamma)          #get a better policy using the value function of the previous one just calculated 
            
            t += 1
            print(t)
            self.Pplot = self.policy.copy()  #keep track of the policy evolution
            self.Vplot = V.copy() # accounting for GUI

            unchanged = True
            for raised in range(4):
                for s in range(self.POSSIBLE_STATES):
                    for a in range(4):
                        if (old_pi[raised][s][a] != self.policy[player_id][raised][s][a]):
                            unchanged = False 
            if unchanged: # normally should be if unchanged
                break
        print('converged after %d iterations' %t) #keep track of the number of (outer) iterations to converge
        self.evaluated[player_id] = True
        return V, self.policy
    
    def valueIteration(self, gamma = 1.0, epsilon = 1e-10):
        print('WIP')


    def get_state(self, player_id):
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


    def get_state_Sim(self, hand, table):
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
    
    def compareHands(self, hand1, hand2, table):
        '''Return 1 if hand 1 is higher, -1 if hand 2 is higher, 0 if hands are equal'''
        state1 = self.get_state_Sim(hand1, table)
        state2 = self.get_state_Sim(hand2, table)
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

    def fillInRewardTableRandom(self):
            init_policy = self.policy
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
                state = self.get_state_Sim(hand1, None)
                chanceOfRaisingFirst = 0
                if init_policy[0][0][state][action['call']] != 1:
                    chanceOfRaisingFirst = init_policy[0][0][state][action['raise']] / (1 - init_policy[0][0][state][action['check']])
                chanceOfFoldingAfter = 0
                chanceOfCallingAfter = 0
                denom = init_policy[0][0][state][action['call']] + init_policy[0][0][state][action['fold']]
                if denom != 0:
                    chanceOfFoldingAfter = init_policy[0][0][state][action['fold']] / denom
                    chanceOfCallingAfter = init_policy[0][0][state][action['call']] / denom
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
                    state = self.get_state_Sim(hand2, None)
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
                                    Pr2[state][action[action2]] += tokens_in
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
                            state = self.get_state_Sim(hand1, table)
                            for hand2  in {'A', 'K', 'Q', 'J', 'T'}: # adversary
                                win = self.compareHands(hand1, hand2, table) # 1 if p1 has better hand, -1 if p2 has better hand, 0 if same hand
                                for action1 in {'call', 'raise', 'fold', 'check'}: # agent
                                    for action2 in {'call', 'raise', 'fold', 'check'}: # adversary
                                        chanceOfRaisingFirst = 0
                                        if init_policy[0][0][state][action['call']] != 1:
                                            chanceOfRaisingFirst = init_policy[0][0][t_index + 1][action['raise']] / (1 - init_policy[0][0][state][action['check']])
                                        chanceOfFoldingAfter = 0
                                        chanceOfCallingAfter = 0
                                        denom = init_policy[0][t_index + 1][state][action['call']] + init_policy[0][t_index + 1][state][action['fold']]
                                        if denom != 0:
                                            chanceOfFoldingAfter = init_policy[0][t_index + 1][state][action['fold']] / denom
                                            chanceOfCallingAfter = init_policy[0][t_index + 1][state][action['call']] / denom
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
                                state = self.get_state_Sim(hand2, table)
                                win = self.compareHands(hand2, hand1, table) # 1 if p2 has better hand, -1 if p1 has better hand, 0 if same hand
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
                        self.R[self.P1][self.FIRST_ROUND][s][a] = 0.0
                    else:
                        self.R[self.P1][self.FIRST_ROUND][s][a] = Pr1[s][a] / np.abs(Pn1[s][a])

                    if Pn2[s][a] == 0:
                        self.R[self.P2][self.FIRST_ROUND][s][a] = 0.0
                    else:
                        self.R[self.P2][self.FIRST_ROUND][s][a] = Pr2[s][a] / np.abs(Pn2[s][a])

                    for pid in range(2):
                        for raised in range(3):
                            if Pn[pid][raised][s][a] == 0:
                                self.R[pid][raised+1][s][a] = 0.0
                            else:
                                self.R[pid][raised+1][s][a] = Pr[pid][raised][s][a] / np.abs(Pn[pid][raised][s][a])

    def step(self, pid, state):
        s = self.get_state(pid)
        legal_actions = state['raw_legal_actions']
        raised = self.FIRST_ROUND
        if s%4 != 0: # if in second round
            raised += int(np.ceil(self.env.game.players[pid].in_chips)) # determine how many chips are raised

        action_name = ['call', 'raise', 'fold', 'check']
        pol = self.policy[pid][raised][s]
        if pid == 0: # p1 case
            if 'check' in legal_actions: # before p2's turn - legal actions are raise, fold, check
                for action in range(len(pol)):
                    if pol[action] == 0.55 or pol[action] == 1:
                        return action_name[action]
            else: # after p2 has raised - legal actions are call, fold
                for action in range(len(pol)):
                    if pol[action] == 0.45 or pol[action] == 1:
                        return action_name[action]
        else: # p2 case
            if 'check' in legal_actions: # p1 has checked - legal actions are raise, fold, check
                for action in range(len(pol)):
                    if pol[action] == 0.55 or pol[action] == 1:
                        return action_name[action]
            else: # p1 has raised - legal actions are call, raise, fold
                for action in range(len(pol)):
                    if pol[action] == 0.45 or pol[action] == 1:
                        return action_name[action]
        # the player cases could perhaps be fused into one, but this might be less confusing?

        # TODO CHANGE PID IF P2 RAISES IN FIRST ROUND
        # TODO THIS ALSO NEEDS TO BE REFLECTED IN THE POLICY

    def eval_step(self, pid, state):
        return self.step(pid, state)