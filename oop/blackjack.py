'''
This script has all the functionalities to play blackjack
'''
from blackjack_player import *
from utilities import *
from deck import *


class Blackjack(object):

	def __init__():
		self.dealer = Dealer()
		self.player = Player()
		self.deck = Deck()
		self.play()


	def play(self):
		while self.player.money > 0:
			self.play_one_round()


	def play_one_round(self):
		bet_size = self.player.bet()
		self.deck.shuffle()

		for i in xrange(2):
			deck = self.player.draw_card(deck)
			deck = self.dealer.draw_card(deck)

		if is_blackjack(self.player.hand):
			self.player.win(1.5 * bet_size)
			return

		#Keep giving the choice to player to hit or stand
		while True:
			if value(player.hand) >= 21:
				break
			if not self.player.hit_or_stand(self.dealer.hand[0]):
				break
			deck = self.player.draw_card(deck)

		player_value = value(self.player.hand)
		if player_value > 21:
			self.player.lose(bet_size)
			return

		#Hit the dealer until at or over 17
		while True:
			if value(dealer.hand) >= 21:
				break
			if not self.dealer.hit():
				break
			deck = self.dealer.draw_card(deck)

		dealer_value = value(self.dealer.hand)
		if player_value > dealer_value:
			self.player.win(bet_size)
		elif dealer_value > player_value:
			self.player.lose(bet_size)
		else:
			print "Push!"


if __name__ == '__main__':
	blackjack = Blackjack()
	blackjack.play()


