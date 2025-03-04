#from rlcard.utils.utils import init_standard_deck
from rlcard.games.base import Card

def init_standard_deck():
    suit_list = ['S', 'H', 'D', 'C']
    rank_list = ['A', 'T', 'J', 'Q', 'K']
    res = [Card(suit, rank) for suit in suit_list for rank in rank_list]
    return res

class LimitHoldemDealer:
    def __init__(self, np_random):
        self.np_random = np_random
        self.deck = init_standard_deck()
        self.shuffle()
        self.pot = 0

    def shuffle(self):
        self.np_random.shuffle(self.deck)

    def deal_card(self):
        """
        Deal one card from the deck

        Returns:
            (Card): The drawn card from the deck
        """
        return self.deck.pop()
