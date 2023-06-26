''' Limit Hold 'em rule model
'''
import rlcard
from rlcard.models.model import Model

class LimitholdemRuleAgentV1(object):
    ''' Limit Hold 'em Rule agent version 1
    '''

    def __init__(self):
        self.use_raw = True

    @staticmethod
    def step(state):
        ''' Predict the action when given raw state. A simple rule-based AI.
        Args:
            state (dict): Raw state from the game

        Returns:
            action (str): Predicted action
        '''
        legal_actions = state['raw_legal_actions']
        state = state['raw_obs']
        hand = state['hand']
        public_cards = state['public_cards']
        action = 'check'
        
        if len(public_cards) == 0:
            if hand[0][1] in ['A', 'K']:
                action = 'raise'
            #elif hand[0][1] in ['Q', 'J']:
            #    action = 'check'
            #elif hand[0][1] in ['T']:
            #    action = 'fold'
                
        if len(public_cards) == 2:
            table = [elem[1] for elem in public_cards]
            if hand[0][1] in table:
                if table[0] == table[1] or hand[0][1] in ['A', 'K']:
                    action = 'raise'
                else:
                    action = 'call'


        #return action
        if action in legal_actions:
            return action
        else:
            if action == 'raise':
                return 'call'
            if action == 'check':
                return 'fold'
            if action == 'call':
                return 'raise'
            else:
                return action

    def eval_step(self, state):
        ''' Step for evaluation. The same to step
        '''
        return self.step(state), []

class LimitholdemRuleModelV1(Model):
    ''' Limitholdem Rule Model version 1
    '''

    def __init__(self):
        ''' Load pretrained model
        '''
        env = rlcard.make('limit-holdem')

        rule_agent = LimitholdemRuleAgentV1()
        self.rule_agents = [rule_agent for _ in range(env.num_players)]

    @property
    def agents(self):
        ''' Get a list of agents for each position in a the game

        Returns:
            agents (list): A list of agents

        Note: Each agent should be just like RL agent with step and eval_step
              functioning well.
        '''
        return self.rule_agents

    @property
    def use_raw(self):
        ''' Indicate whether use raw state and action

        Returns:
            use_raw (boolean): True if using raw state and action
        '''
        return True
