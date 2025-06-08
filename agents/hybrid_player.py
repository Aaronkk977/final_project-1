import random
import torch
from game.players import BasePokerPlayer
from solver import best_of_seven, hand_rank

class HybridPlayer(BasePokerPlayer):
    def __init__(self):
        self.preflop_table = self.load_preflop_csv("preflop_equity_2000.csv")

    def load_preflop_ranges(self, path):
        ranges = {}
        with open(path) as f:
            for line in f:
                card_combo, freq = line.strip().split(',')
                ranges[tuple(card_combo.split('-'))] = float(freq)
        return ranges

    def estimate_winrate_mc(self, hole_card, community, N=2000):
        # Monte Carlo 簡易實作：隨機模擬對手手牌＋未來公共牌
        wins = ties = losses = 0
        deck_template = Deck().full_deck()  # 假設有 Deck 類別
        known = hole_card + community
        for _ in range(N):
            deck = [c for c in deck_template if c not in known]
            opp = random.sample(deck, 2)
            deck2 = [c for c in deck if c not in opp]
            draw = random.sample(deck2, 5 - len(community))
            board = community + draw
            my_best = best_of_seven(hole_card, board)
            opp_best = best_of_seven(opp, board)
            if my_best > opp_best: wins += 1
            elif my_best == opp_best: ties += 1
            else: losses += 1
        return wins / N, ties / N, losses / N

    def estimate_winrate_ml(self, hole_card, community, features_fn):
        # ML 推論範例：把 features 轉成 tensor，輸出勝率
        features = features_fn(hole_card, community)
        with torch.no_grad():
            win_rate = self.model(features).item()
        return win_rate

    def declare_action(self, valid_actions, hole_card, round_state):
        street = round_state["street"]
        community = round_state["community_card"]
        # Preflop
        if street == "preflop":
            print(f"Preflop hole_card: {hole_card}, community: {community}")
            hand = hole_card
            winrate = self.preflop_table.get(hand, 0)
            if winrate > 0.7:
                return valid_actions[2]["action"], valid_actions[2]["amount"]["min"] # raise min
            elif winrate > 0.4:
                return valid_actions[1]["action"], valid_actions[1]["amount"] # call
            else:
                return valid_actions[0]["action"], 0 # fold
            
        # Post-flop (flop)
        elif street == "flop":
            win_mc, _, _ = self.estimate_winrate_mc(hole_card, community)
            if win_mc > 0.6:
                act, amt = valid_actions[2]["action"], valid_actions[2]["amount"]["min"]
            elif win_mc > 0.3:
                act, amt = valid_actions[1]["action"], valid_actions[1]["amount"]
            else:
                act, amt = valid_actions[0]["action"], 0
                
        # Turn/River
        else:  # turn or river
            # 簡單 rule-based：若有強順子以上就全下，否則保守跟注
            best = best_of_seven(hole_card, community)
            if best >= 5:
                act, amt = valid_actions[2]["action"], valid_actions[2]["amount"]["max"]
            else:
                act, amt = valid_actions[1]["action"], valid_actions[1]["amount"]

        return act, amt

    # 其餘 callback 留空或做 logging
    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

def setup_ai():
    return HybridPlayer()
