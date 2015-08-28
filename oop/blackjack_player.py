'''
This script has all the functionalities of individuals related to blackjack
'''
from utilities import *

class Individual(object):

	def __init__(self):
		self.hand = []


	def draw_card(self, deck):
		self.hand.append(deck.draw_card())
		return deck


	def show_hand(self):
		print_hand(self.hand)


class Player(Individual):

	def __init__(self):
		self.name = raw_input('Enter your name: ')
		self.money = int(raw_input('Enter the amount of money you have: '))
		self.hand = []


	def bet(self):
		bet_size = 0
		while not (bet_size > 0 and bet_size <= self.money):
			bet_size = input('You have %d amount of money. How much would you like to bet: ' % self.money)
		return bet_size


	def win_money(self, amount):
		print 'You just won %d units of money' % amount
		self.money += amount


	def lose_money(self, amount):
		print 'You just lost %d units of money' % amount
		self.money -= amount


	def hit_or_stand(self, visible_dealer_card):
		print 'You have %s' % print_hand(self.hand)
		print 'Your dealer has %s' % str(visible_dealer_card)
		choice = None
		while choice not in ("h", "s"):
			choice = raw_input("What would you like to do? Hit or stand? Type 'h' \
				for hit and 's' for stand: ")
		return choice == "h"


class Dealer(Individual):

	def hit(self):
		return hand_value(self.hand) < 17


