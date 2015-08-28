'''
This script tests the functionalities of blackjack_player and blackjack
'''

import nose.tools as n
from blackjack_player import *
from blackjack import *
from utilities import *

def test_value():
	cards = [Card('5','s'), Card('Q','d')]
	n.assert_equal(value(cards), 15)


def test_player_draw_card():
	player = Player()
	deck = Deck()
	n.assert_equal(len(deck), 52)
	card = deck[50] #last card in deck
	deck = player.draw_card(deck)
	n.assert_equal(len(deck), 51)
	n.assert_equal(player.hand[0], card)