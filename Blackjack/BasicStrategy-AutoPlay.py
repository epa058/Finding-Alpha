import random

class Blackjack:
    def __init__(self, num_decks=8):
        # Cards and stuff
        self.ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        self.card_values = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            '10': 10, 'J': 10, 'Q': 10, 'K': 10, 'A': 11
        }
        self.num_decks = num_decks
        self.deck = []
        self.player_hand = []
        self.dealer_hand = []

        # Counter
        self.hands_played = 0
        self.player_wins = 0

        # Basic Strategy
        '''
        https://m.espacejeux.com/en/casino/game/table-games/blackjack-gol
        https://www.blackjackinfo.com/blackjack-basic-strategy-engine/?numdecks=8&soft17=s17&dbl=all&das=yes&surr=ns&peek=no
        '''
        # H=Hit, S=Stand, D=Double Down if allowed else Hit, DS=Double Down if allowed else Stand, P=Split
        self.strategy_hard = {
            21: {v: 'S' for v in range(2, 12)},
            20: {v: 'S' for v in range(2, 12)},
            19: {v: 'S' for v in range(2, 12)},
            18: {v: 'S' for v in range(2, 12)},
            17: {v: 'S' for v in range(2, 12)},
            16: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            15: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            14: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            13: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            12: {2: 'H', 3: 'H', 4: 'S', 5: 'S', 6: 'S', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            11: {2: 'D', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'D', 8: 'D', 9: 'D', 10: 'H', 11: 'H'},
            10: {2: 'D', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'D', 8: 'D', 9: 'D', 10: 'H', 11: 'H'},
            9:  {2: 'H', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            8:  {v: 'H' for v in range(2, 12)},
            7: {v: 'H' for v in range(2, 12)},
            6:  {v: 'H' for v in range(2, 12)},
            5: {v: 'H' for v in range(2, 12)},
            4: {v: 'H' for v in range(2, 12)},
        }
        self.strategy_soft = {
            # Ace is counted as 11
            21: {v: 'S' for v in range(2, 12)},
            20: {v: 'S' for v in range(2, 12)},
            19: {v: 'S' for v in range(2, 12)},
            18: {2: 'S', 3: 'DS', 4: 'DS', 5: 'DS', 6: 'DS', 7: 'S', 8: 'S', 9: 'H', 10: 'H', 11: 'H'},
            17: {2: 'H', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            16: {2: 'H', 3: 'H', 4: 'D', 5: 'D', 6: 'D', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            15: {2: 'H', 3: 'H', 4: 'D', 5: 'D', 6: 'D', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            14: {2: 'H', 3: 'H', 4: 'H', 5: 'D', 6: 'D', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            13: {2: 'H', 3: 'H', 4: 'H', 5: 'D', 6: 'D', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
        }
        self.strategy_pairs = {
            'A': {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 11: 'H'}, 
            10: {v: 'S' for v in range(2, 12)},
            9: {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'S', 8: 'P', 9: 'P', 10: 'S', 11: 'S'},
            8: {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'P', 8: 'P', 9: 'P', 10: 'H', 11: 'H'}, 
            7: {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'P', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            6: {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            5: {2: 'D', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'D', 8: 'D', 9: 'D', 10: 'H', 11: 'H'},
            4: {2: 'H', 3: 'H', 4: 'H', 5: 'P', 6: 'P', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            3: {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'P', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            2: {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'P', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
        }

    def create_deck(self):
        self.deck = [{'rank': rank, 'suit': suit} for _ in range(self.num_decks) for rank in self.ranks for suit in self.suits]
        self.shuffle_deck()
        
    def shuffle_deck(self):
        random.shuffle(self.deck)

    def deal_card(self):
        if not self.deck or len(self.deck) < 52:
            print("--- Reshuffling the shoe ---")
            self.create_deck()
        return self.deck.pop()

    def get_hand_value(self, hand):
        """
        Calculates the value of a hand, handles Aces correctly.
        is_soft is True if an Ace is counted as 11.
        """
        value = sum(self.card_values[card['rank']] for card in hand)
        num_aces = sum(1 for card in hand if card['rank'] == 'A')
        while value > 21 and num_aces > 0:
            value -= 10
            num_aces -= 1

        is_soft = num_aces > 0 and value <= 21
        return value, is_soft

    def is_pair(self, hand):
        return len(hand) == 2 and hand[0]['rank'] == hand[1]['rank']

    def get_basic_strategy_move(self, player_hand, dealer_upcard, can_double_down, allow_split=True):
        player_value, is_soft = self.get_hand_value(player_hand)
        dealer_value = self.card_values[dealer_upcard['rank']]
        
        move = "Error"

        if allow_split and self.is_pair(player_hand):
            pair_rank = player_hand[0]['rank']
            if pair_rank in ['J', 'Q', 'K', '10']:
                move = self.strategy_pairs[10][dealer_value]
            elif pair_rank == 'A':
                move = self.strategy_pairs['A'][dealer_value]
            else:
                move = self.strategy_pairs[int(pair_rank)][dealer_value]

        elif is_soft:
            move = self.strategy_soft.get(player_value, self.strategy_hard.get(player_value, 'H'))[dealer_value]

        else: # Hard total
            if player_value > 21: return 'S' 
            move = self.strategy_hard.get(player_value, 'S' if player_value > 16 else 'H')[dealer_value]

        if move == 'D' and not can_double_down: return 'H'
        if move == 'DS' and not can_double_down: return 'S'

        return move
    
    def _format_hand(self, hand):
        """Helper function to create a nice string representation of a hand."""
        return ", ".join([f"{card['rank']}" for card in hand])

    def display_stats(self):
        print(f"\n--- Stats ---")
        win_rate = (self.player_wins / self.hands_played) * 100 if self.hands_played > 0 else 0
        print(f"Player Wins: {self.player_wins} / {self.hands_played} (Win Rate: {win_rate:.2f}%)")
        print(f"-------------")

    def play_hand(self):
        self.hands_played += 1
        print("-" * 30)
        print(f"Hand #{self.hands_played}")

        player_hand, dealer_hand = [], []
        player_hand.append(self.deal_card())
        dealer_hand.append(self.deal_card())
        player_hand.append(self.deal_card())
        dealer_hand.append(self.deal_card())
        
        dealer_upcard = dealer_hand[0]
        player_score, _ = self.get_hand_value(player_hand)
        dealer_score, _ = self.get_hand_value(dealer_hand)

        print(f"Player's Hand: [{self._format_hand(player_hand)}] ({player_score})")
        print(f"Dealer's Upcard: [{dealer_upcard['rank']}]")

        # Check for Blackjack
        if player_score == 21:
            print("Blackjack!")
            if dealer_score != 21:
                self.player_wins += 1
                print("Player wins!")
            else:
                print("Push.")
            self.display_stats()
            return

        # Player's Turn
        player_hands = [player_hand]
        final_player_scores = []
        i = 0
        num_splits = 0
        has_split_aces = False

        while i < len(player_hands):
            current_hand = player_hands[i]
            hand_label = f"Hand {i+1}" if (len(player_hands) > 1) else "Player"
            initial_move = self.get_basic_strategy_move(current_hand, dealer_upcard, True, allow_split=True)
            
            if initial_move == 'P':
                is_ace_pair = current_hand[0]['rank'] == 'A'

                if is_ace_pair and not has_split_aces:
                    print(f"  > Splitting Aces...")
                    has_split_aces = True
                    self.hands_played += 1
                    hand1 = [current_hand[0], self.deal_card()]
                    hand2 = [current_hand[1], self.deal_card()]
                    final_player_scores.append(self.get_hand_value(hand1)[0])
                    final_player_scores.append(self.get_hand_value(hand2)[0])
                    print(f"    Ace Hand 1 final: [{self._format_hand(hand1)}] ({final_player_scores[-2]})")
                    print(f"    Ace Hand 2 final: [{self._format_hand(hand2)}] ({final_player_scores[-1]})")
                    player_hands.pop(i)
                    continue 

                elif not is_ace_pair and len(player_hands) < 4:
                    print(f"  > Splitting {current_hand[0]['rank']}s...")
                    self.hands_played += 1
                    player_hands.pop(i)
                    hand1 = [current_hand[0], self.deal_card()]
                    hand2 = [current_hand[1], self.deal_card()]
                    print(f"    New Hand created: [{self._format_hand(hand1)}]")
                    print(f"    New Hand created: [{self._format_hand(hand2)}]")
                    player_hands.insert(i, hand2) 
                    player_hands.insert(i, hand1)
                    continue

            while True: 
                player_score, _ = self.get_hand_value(current_hand)
                if player_score >= 21:
                    break
                
                can_double_down = len(current_hand) == 2
                move = self.get_basic_strategy_move(current_hand, dealer_upcard, can_double_down, allow_split=False)
            
                print(f" > {hand_label} [{self._format_hand(current_hand)}] ({player_score}) strategy: {move}")
                
                if move == 'S':
                    break
                
                if move in ['D', 'DS']:
                    current_hand.append(self.deal_card())
                    print(f"    {hand_label} doubles down, gets {current_hand[-1]['rank']}.")
                    break

                if move == 'H':
                    current_hand.append(self.deal_card())
                    print(f"    {hand_label} hits: [{self._format_hand(current_hand)}] ({self.get_hand_value(current_hand)[0]})")
            
            final_score = self.get_hand_value(current_hand)[0]
            print(f" > {hand_label} final score: [{self._format_hand(current_hand)}] ({final_score})")
            final_player_scores.append(final_score)
            i += 1

        # --- Dealer's Turn ---
        print("\nDealer's turn...")
        dealer_score, _ = self.get_hand_value(dealer_hand)
        print(f"Dealer's Hand: [{self._format_hand(dealer_hand)}] ({dealer_score})")
        while dealer_score < 17:
            dealer_hand.append(self.deal_card())
            dealer_score, _ = self.get_hand_value(dealer_hand)
            print(f"Dealer hits: [{self._format_hand(dealer_hand)}] ({dealer_score})")


        # Results
        print("\n--- Results ---")
        print(f"Dealer has: {dealer_score}")
        for j, p_score in enumerate(final_player_scores):
            hand_label = f"Hand {j+1}" if len(final_player_scores) > 1 else "Player"
            if p_score > 21:
                print(f"{hand_label} ({p_score}) busts. Dealer wins.")
            elif dealer_score > 21 or p_score > dealer_score:
                print(f"{hand_label} ({p_score}) wins!")
                self.player_wins += 1
            elif p_score < dealer_score:
                print(f"{hand_label} ({p_score}) loses. Dealer wins.")
            else:
                print(f"{hand_label} ({p_score}) pushes.")
        
        self.display_stats()

# Main execution block
if __name__ == "__main__":
    # Create an instance of our Blackjack engine
    blackjack_bot = Blackjack(num_decks=8)
    
# Play
    for _ in range(100):
        blackjack_bot.play_hand()
