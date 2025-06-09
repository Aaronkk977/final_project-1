from functools import reduce
from itertools import groupby
from itertools import combinations # New import for generating 5-card combinations

class HandEvaluator:

    HIGHCARD = 0
    ONEPAIR = 1 << 8
    TWOPAIR = 1 << 9
    THREECARD = 1 << 10
    STRAIGHT = 1 << 11
    FLASH = 1 << 12
    FULLHOUSE = 1 << 13
    FOURCARD = 1 << 14
    STRAIGHTFLASH = 1 << 15

    HAND_STRENGTH_MAP = {
        HIGHCARD: "HIGHCARD",
        ONEPAIR: "ONEPAIR",
        TWOPAIR: "TWOPAIR",
        THREECARD: "THREECARD",
        STRAIGHT: "STRAIGHT",
        FLASH: "FLASH",
        FULLHOUSE: "FULLHOUSE",
        FOURCARD: "FOURCARD",
        STRAIGHTFLASH: "STRAIGHTFLASH",
    }

    @classmethod
    def gen_hand_rank_info(self, hole, community):
        hand_score = self.eval_hand(hole, community)

        # Extract hand type (H)
        actual_hand_strength_type = self.__mask_hand_strength(hand_score)
        strength_str = self.HAND_STRENGTH_MAP[actual_hand_strength_type]

        # Extract primary ranks (R1, R2) for the hand
        hand_high_rank = self.__mask_hand_high_rank(hand_score)
        hand_low_rank = self.__mask_hand_low_rank(hand_score)

        # Get hole card ranks directly from the input "hole" cards
        # Sort by rank, highest first. Default to 0 if no/few cards.
        hole_card_1_rank = 0
        hole_card_2_rank = 0
        if len(hole) >= 1:
            sorted_hole_cards = sorted(hole, key=lambda card: card.rank, reverse=True)
            hole_card_1_rank = sorted_hole_cards[0].rank
            if len(hole) >= 2:
                hole_card_2_rank = sorted_hole_cards[1].rank

        return {
            "hand": {"strength": strength_str, "high": hand_high_rank, "low": hand_low_rank},
            "hole": {"high": hole_card_1_rank, "low": hole_card_2_rank},
        }

    @classmethod
    def eval_hand(self, hole, community):
        """
        Original logic for eval_hand
        """
        # ranks = sorted([card.rank for card in hole])
        # hole_flg = ranks[1] << 4 | ranks[0]
        # hand_flg = self.__calc_hand_info_flg(hole, community) << 8
        # return hand_flg | hole_flg
        
        """
        Updated logic for eval_hand
        """
        all_cards = hole + community
        best_combo_val = None
        for combo_tuple in combinations(all_cards, 5):
            combo = list(combo_tuple)
            # Use the same logic as __calc_hand_info_flg, but only for this 5-card combo, hand_strength_and_primary_ranks will be like (TWOPAIR | (rank_H << 4 | rank_L)) or HIGHCARD
            hand_strength_and_primary_ranks = self.__calc_hand_info_flg([], combo)
            
            # Shift to make space for 5 kickers (4 bits each = 20 bits)
            current_eval_score = hand_strength_and_primary_ranks << 20

            ranks = sorted([card.rank for card in combo], reverse=True)
            for i in range(5):
                current_eval_score |= ranks[i] << (4 * (4 - i)) # Add 5 kickers

            if best_combo_val is None or current_eval_score > best_combo_val:
                best_combo_val = current_eval_score
        return best_combo_val

    # Return Format
    # [Bit flg of hand][rank1(4bit)][rank2(4bit)]
    # ex.)
    #       HighCard hole card 3,4   =>           100 0011
    #       OnePair of rank 3        =>        1 0011 0000
    #       TwoPair of rank A, 4     =>       10 1110 0100
    #       ThreeCard of rank 9      =>      100 1001 0000
    #       Straight of rank 10      =>     1000 1010 0000
    #       Flash of rank 5          =>    10000 0101 0000
    #       FullHouse of rank 3, 4   =>   100000 0011 0100
    #       FourCard of rank 2       =>  1000000 0010 0000
    #       straight flash of rank 7 => 10000000 0111 0000
    @classmethod
    def __calc_hand_info_flg(self, hole, community):
        cards = hole + community
        if self.__is_straightflash(cards):
            return self.STRAIGHTFLASH | self.__eval_straightflash(cards)
        if self.__is_fourcard(cards):
            return self.FOURCARD | self.__eval_fourcard(cards)
        if self.__is_fullhouse(cards):
            return self.FULLHOUSE | self.__eval_fullhouse(cards)
        if self.__is_flash(cards):
            return self.FLASH | self.__eval_flash(cards)
        if self.__is_straight(cards):
            return self.STRAIGHT | self.__eval_straight(cards)
        if self.__is_threecard(cards):
            return self.THREECARD | self.__eval_threecard(cards)
        if self.__is_twopair(cards):
            return self.TWOPAIR | self.__eval_twopair(cards)
        if self.__is_onepair(cards):
            return self.ONEPAIR | (self.__eval_onepair(cards))
        
        """
        If called from eval_hand's loop (hole=[]), and it's a 5-card high-card hand, its strength is HIGHCARD (0). Kicker ranks are handled by eval_hand.
        If called with actual hole cards (not from eval_hand's loop), this path means hole cards + community make a high card hand based on hole cards.
        """
        if not hole: # evaluating a 5-card combo from eval_hand
            return self.HIGHCARD
        else: # Original fallback for __calc_hand_info_flg when hole cards are present
            ranks = sorted([card.rank for card in hole])
            return self.HIGHCARD | (ranks[1] << 4 | ranks[0]) if len(ranks) == 2 else self.HIGHCARD

    @classmethod
    def __is_onepair(self, cards):
        return self.__eval_onepair(cards) != 0

    @classmethod
    def __eval_onepair(self, cards):
        rank = 0
        memo = 0  # bit memo
        for card in cards:
            mask = 1 << card.rank
            if memo & mask != 0:
                rank = max(rank, card.rank)
            memo |= mask
        return rank << 4

    @classmethod
    def __is_twopair(self, cards):
        return len(self.__search_twopair(cards)) == 2

    @classmethod
    def __eval_twopair(self, cards):
        ranks = self.__search_twopair(cards)
        return ranks[0] << 4 | ranks[1]

    @classmethod
    def __search_twopair(self, cards):
        ranks = []
        memo = 0
        for card in cards:
            mask = 1 << card.rank
            if memo & mask != 0:
                ranks.append(card.rank)
            memo |= mask
        return sorted(ranks)[::-1][:2]

    @classmethod
    def __is_threecard(self, cards):
        return self.__search_threecard(cards) != -1

    @classmethod
    def __eval_threecard(self, cards):
        return self.__search_threecard(cards) << 4

    @classmethod
    def __search_threecard(self, cards):
        rank = -1
        bit_memo = reduce(
            lambda memo, card: memo + (1 << (card.rank - 1) * 3), cards, 0
        )
        for r in range(2, 15):
            bit_memo >>= 3
            count = bit_memo & 7
            if count >= 3:
                rank = r
        return rank

    @classmethod
    def __is_straight(self, cards):
        return self.__search_straight(cards) != -1

    @classmethod
    def __eval_straight(self, cards):
        return self.__search_straight(cards) << 4

    @classmethod
    def __search_straight(self, cards):
        # Create a bitmask of ranks present.
        bit_memo = 0
        for card in cards:
            bit_memo |= (1 << card.rank)

        # Check for standard straights (T-A down to 6-2)
        for high_rank_val in range(14, 5, -1): # high_rank_val from 14 down to 6
            is_this_straight = True
            for i in range(5): # check from high_rank_val to high_rank_val - 4
                if not (bit_memo & (1 << (high_rank_val - i))):
                    is_this_straight = False
                    break
            if is_this_straight:
                return high_rank_val # Return the highest card's rank

        # Check for A,2,3,4,5 straight (wheel)
        if (bit_memo & (1<<14)) and \
           (bit_memo & (1<<2)) and \
           (bit_memo & (1<<3)) and \
           (bit_memo & (1<<4)) and \
           (bit_memo & (1<<5)):
            return 5

        return -1 # No straight

    @classmethod
    def __is_flash(self, cards):
        return self.__search_flash(cards) != -1

    @classmethod
    def __eval_flash(self, cards):
        return self.__search_flash(cards) << 4

    @classmethod
    def __search_flash(self, cards):
        best_suit_rank = -1
        fetch_suit = lambda card: card.suit
        fetch_rank = lambda card: card.rank
        for suit, group_obj in groupby(sorted(cards, key=fetch_suit), key=fetch_suit):
            g = list(group_obj)
            if len(g) >= 5:
                max_rank_card = max(g, key=fetch_rank)
                best_suit_rank = max(best_suit_rank, max_rank_card.rank)
        return best_suit_rank

    @classmethod
    def __is_fullhouse(self, cards):
        r1, r2 = self.__search_fullhouse(cards)
        return r1 and r2

    @classmethod
    def __eval_fullhouse(self, cards):
        r1, r2 = self.__search_fullhouse(cards)
        return r1 << 4 | r2

    @classmethod
    def __search_fullhouse(self, cards):
        fetch_rank = lambda card: card.rank
        three_card_ranks, two_pair_ranks = [], []
        for rank, group_obj in groupby(sorted(cards, key=fetch_rank), key=fetch_rank):
            g = list(group_obj)
            if len(g) >= 3:
                three_card_ranks.append(rank)
            if len(g) >= 2:
                two_pair_ranks.append(rank)
        two_pair_ranks = [
            rank for rank in two_pair_ranks if not rank in three_card_ranks
        ]
        if len(three_card_ranks) == 2:
            two_pair_ranks.append(min(three_card_ranks))
        max_ = lambda l: None if len(l) == 0 else max(l)
        return max_(three_card_ranks), max_(two_pair_ranks)

    @classmethod
    def __is_fourcard(self, cards):
        return self.__eval_fourcard(cards) != 0

    @classmethod
    def __eval_fourcard(self, cards):
        rank = self.__search_fourcard(cards)
        return rank << 4

    @classmethod
    def __search_fourcard(self, cards):
        fetch_rank = lambda card: card.rank
        for rank, group_obj in groupby(sorted(cards, key=fetch_rank), key=fetch_rank):
            g = list(group_obj)
            if len(g) >= 4:
                return rank
        return 0

    @classmethod
    def __is_straightflash(self, cards):
        return self.__search_straightflash(cards) != -1

    @classmethod
    def __eval_straightflash(self, cards):
        return self.__search_straightflash(cards) << 4

    @classmethod
    def __search_straightflash(self, cards):
        flash_cards = []
        fetch_suit = lambda card: card.suit
        for suit, group_obj in groupby(sorted(cards, key=fetch_suit), key=fetch_suit):
            g = list(group_obj)
            if len(g) >= 5:
                flash_cards = g
        return self.__search_straight(flash_cards)

    @classmethod
    def __mask_hand_strength(self, bit):
        # Score format from eval_hand (with diff applied): ( (H | R) << 20 ) | KICKERS_20BIT
        # H = Hand Type (e.g. ONEPAIR), R = Primary Ranks (8-bit)
        # We need to extract H.
        val_H_R = bit >> 20  # This is (H | R)
        # H is in bits 8 and above of (H|R). R is in bits 0-7.
        # H values are 0, 1<<8, ..., 1<<15. Mask out R part.
        return val_H_R & (~0xFF)

    @classmethod
    def __mask_hand_high_rank(self, bit):
        # Extracts the higher primary rank from R in ( (H | R) << 20 )
        val_H_R = bit >> 20
        R_part = val_H_R & 0xFF  # R = rank1 << 4 | rank2
        return (R_part >> 4) & 0xF # rank1

    @classmethod
    def __mask_hand_low_rank(self, bit):
        # Extracts the lower primary rank from R in ( (H | R) << 20 )
        val_H_R = bit >> 20
        R_part = val_H_R & 0xFF  # R = rank1 << 4 | rank2
        return R_part & 0xF      # rank2

    @classmethod
    def __mask_hole_high_rank(self, bit):
        # This function is no longer suitable for getting hole card ranks from the new score format.
        # Hole ranks should be obtained directly from the 'hole' cards parameter.
        raise NotImplementedError("Hole ranks should be derived directly from hole cards, not the combined score.")

    @classmethod
    def __mask_hole_low_rank(self, bit):
        raise NotImplementedError("Hole ranks should be derived directly from hole cards, not the combined score.")