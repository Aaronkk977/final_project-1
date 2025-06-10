import random

class Deck:
    ranks = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
    suits = ['H','D','C','S']

    @classmethod
    def full_deck(cls):
        """
        回傳 52 張牌的列表，例如 ['2h','2d',...,'As']。
        """
        return [s + r for r in cls.ranks for s in cls.suits]

    @classmethod
    def shuffled_deck(cls):
        """
        回傳一副已經隨機打亂的牌堆。
        """
        deck = cls.full_deck()
        random.shuffle(deck)
        return deck

    @classmethod
    def draw(cls, deck, n=1):
        """
        從傳入的列表 deck 前面抽 n 張，回傳 (抽出的牌列表, 剩餘的牌堆)。
        """
        return deck[:n], deck[n:]
    
    @classmethod
    def remove(cls, deck, cards):
        """
        從傳入的列表 deck 中移除 cards 中的牌，回傳新的牌堆。
        cards 可以是一張牌的字串，也可以是一個字串列表。
        """
        # 統一把單張轉成列表
        if isinstance(cards, str):
            cards = [cards]
        # 過濾
        return [c for c in deck if c not in cards]
