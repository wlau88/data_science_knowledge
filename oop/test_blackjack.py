'''
This script tests the functionalities of blackjack_player and blackjack
'''

import nose.tools as n
from blackjack_player import *
from blackjack import *
from utilities import *

def test_hand_value():
	cards = [Card('5','s'), Card('Q','d')]
	n.assert_equal(hand_value(cards), 15)


def test_individual_draw_card():
	individual = Individual()
	deck = Deck()
	n.assert_equal(len(deck), 52)
	deck = individual.draw_card(deck)
	n.assert_equal(len(deck), 51)