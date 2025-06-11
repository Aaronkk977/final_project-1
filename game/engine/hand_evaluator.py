from functools import reduce
from itertools import groupby


class HandEvaluator:

    HIGHCARD = 0
    ONEPAIR = 1 << 20
    TWOPAIR = 1 << 21
    THREECARD = 1 << 22
    STRAIGHT = 1 << 23
    FLASH = 1 << 24
    FULLHOUSE = 1 << 25
    FOURCARD = 1 << 26
    STRAIGHTFLASH = 1 << 27

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
        hand = self.eval_hand(hole, community)
        row_strength = self.__mask_hand_strength(hand)
        strength = self.HAND_STRENGTH_MAP[row_strength]
        hand_rank_1 = self.__mask_hand_rank_1(hand)
        hand_rank_2 = self.__mask_hand_rank_2(hand)
        hand_rank_3 = self.__mask_hand_rank_3(hand)
        hand_rank_4 = self.__mask_hand_rank_4(hand)
        hand_rank_5 = self.__mask_hand_rank_5(hand)

        return {
            "hand": {"strength": strength, 
                     "rank_1": hand_rank_1,
                     "rank_2": hand_rank_2, 
                     "rank_3": hand_rank_3,
                     "rank_4": hand_rank_4, 
                     "rank_5": hand_rank_5},
        }

    @classmethod
    def eval_hand(self, hole, community):
        ranks = sorted([card.rank for card in hole])
        # hole_flg = ranks[1] << 4 | ranks[0] # we now do not need to consider hole cards separately
        hand_flg = self.__calc_hand_info_flg(hole, community)
        return hand_flg

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
            return self.ONEPAIR | self.__eval_onepair(cards)
        return self.HIGHCARD | self.__eval_highcard(cards)

    @classmethod
    def __eval_highcard(self, cards):
        ranks = sorted([card.rank for card in cards], reverse=True)[:5]
        result = 0
        for i, rank in enumerate(ranks):
            result |= rank << (16 - 4 * i)
        return result

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
            
        kickers = [card.rank for card in cards if card.rank != rank]
        kickers = sorted(kickers, reverse=True)[:3]
            
        # [pair_rank << 16] | [kicker1 << 12] | [kicker2 << 8] | [kicker3 << 4]
        result = rank << 16
        for i, kicker in enumerate(kickers):
            result |= kicker << (12 - 4 * i)
        return result

    @classmethod
    def __is_twopair(self, cards):
        return len(self.__search_twopair(cards)) == 2

    @classmethod
    def __eval_twopair(self, cards):
        pair_ranks = self.__search_twopair(cards)
        kickers = [card.rank for card in cards if card.rank not in pair_ranks]
        kicker = max(kickers) if kickers else 0
        # [high_pair << 16] | [low_pair << 12] | [kicker << 8]
        return (pair_ranks[0] << 16) | (pair_ranks[1] << 12) | (kicker << 8)

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
        three_rank = self.__search_threecard(cards)
        kickers = [card.rank for card in cards if card.rank != three_rank]
        kickers = sorted(kickers, reverse=True)[:2]
        # [three_rank << 16] | [kicker1 << 12] | [kicker2 << 8]
        result = three_rank << 16
        for i, kicker in enumerate(kickers):
            result |= kicker << (12 - 4 * i)
        return result

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
        return self.__search_straight(cards) << 16 # 4 -> 16

    @classmethod
    def __search_straight(self, cards):
        bit_memo = reduce(lambda memo, card: memo | 1 << card.rank, cards, 0)
        rank = -1
        straight_check = lambda acc, i: acc & (bit_memo >> (r + i) & 1) == 1
        for r in range(2, 15):
            if reduce(straight_check, range(5), True):
                rank = r + 4  
                
        # Check for special case wheel (A-2-3-4-5)
        if all(bit_memo & (1 << x) for x in [14, 2, 3, 4, 5]):
            rank = 5
        return rank

    @classmethod
    def __is_flash(self, cards):
        return len(self.__search_flash(cards)) == 5 

    @classmethod
    def __eval_flash(self, cards):
        flush_ranks = self.__search_flash(cards)
        result = 0
        for i, rank in enumerate(flush_ranks):
            result |= rank << (16 - 4 * i)
        return result

    @classmethod
    def __search_flash(self, cards):
        suit_groups = {}
        for card in cards:
            suit_groups.setdefault(card.suit, []).append(card)
        best_flush = []
        for group in suit_groups.values():
            if len(group) >= 5:
                flush_cards = sorted(group, key=lambda c: c.rank, reverse=True)[:5]
                if not best_flush or [c.rank for c in flush_cards] > [c.rank for c in best_flush]:
                    best_flush = flush_cards
        return [c.rank for c in best_flush]

    @classmethod
    def __is_fullhouse(self, cards):
        r1, r2 = self.__search_fullhouse(cards)
        return r1 and r2

    @classmethod
    def __eval_fullhouse(self, cards):
        r1, r2 = self.__search_fullhouse(cards)
        return r1 << 16 | r2 << 12  # 4 -> 16, 0 -> 12

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
        four_rank = self.__search_fourcard(cards)
        if four_rank == 0:
            return 0
        kickers = [card.rank for card in cards if card.rank != four_rank]
        kicker = max(kickers) if kickers else 0
        # [four_rank << 16] | [kicker << 12]
        return (four_rank << 16) | (kicker << 12)

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
        return self.__search_straightflash(cards) << 16

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
        mask = 511 << 20
        return (bit & mask)

    @classmethod
    def __mask_hand_rank_1(self, bit):
        mask = 15 << 16
        return (bit & mask) >> 16

    @classmethod
    def __mask_hand_rank_2(self, bit):
        mask = 15 << 12
        return (bit & mask) >> 12

    @classmethod
    def __mask_hand_rank_3(self, bit):
        mask = 15 << 8
        return (bit & mask) >> 8

    @classmethod
    def __mask_hand_rank_4(self, bit):
        mask = 15 << 4
        return (bit & mask) >> 4

    @classmethod
    def __mask_hand_rank_5(self, bit):
        mask = 15
        return bit & mask