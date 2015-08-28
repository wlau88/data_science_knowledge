'''
Utilities for blackjack game
'''

def hand_value(hand):
	total = 0
	ace = False
	for card in hand:
		if card.number.isdigit():
			total += int(card.number)
		if card.number in "JQK":
			total += 10
		else: # Ace case
			total += 1
			ace = True
	# Different calculation for Ace
	if ace and total <= 11:
		return total + 10

	return total


def is_blackjack(hand):
	return len(hand) == 2 and value(hand) == 21


def is_bust(hand):
	return value(hand) > 21


def print_hand(hand):
	return " ".join(str(card) for card in hand)

