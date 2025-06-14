from agents.deck import Deck
import random
import time
import itertools
from collections import Counter

def _technical_fold(self, round_state):
    """
    must win if fold in next rounds
    """
    sb = round_state["small_blind_amount"]
    my_stack = next(s["stack"] for s in round_state["seats"] if s["uuid"] == self.uuid)
    remaining_rounds = 20 - round_state["round_count"] + 1
    if remaining_rounds % 2 == 0:
        if my_stack - 1000 > remaining_rounds * (sb * 3) * 0.5:
            print("[Technical Fold] must win if fold in next rounds")
            return True
        else:
            return False
    else:
        if my_stack - 1000 > (remaining_rounds-1) * (sb * 3) * 0.5 + sb * 2:
            print("[Technical Fold] must win if fold in next rounds")
            return True
        else:
            return False

def _can_fold(self, round_state):
    """
    must win if fold in next rounds
    """
    sb = round_state["small_blind_amount"]
    my_stack = next(s["stack"] for s in round_state["seats"] if s["uuid"] == self.uuid)
    current = round_state["round_count"]
    
    if current == 20 and my_stack < sb * 2 + 1000:
        print("[Cannot Fold] Cannot fold in the last round")
        return False
    return True

def estimate_winrate_mc(self, hole, community, iterations):
    """
    Monte Carlo 模擬勝率，支援快取與超時回退
    回傳 (win_rate, tie_rate, loss_rate)
    """
    wins = ties = 0
    start = time.time()

    for i in range(iterations):
        # 超時檢查：如果超過 time_limit，就提早跳出
        if time.time() - start > self.mc_time_limit:
            break
        # 準備牌堆，移除已知牌
        deck = Deck.remove(self._deck, hole + community)
        # 隨機抽對手洞牌
        opp = random.sample(deck, 2)
        deck = Deck.remove(deck, opp)
        # 補齊公共牌到 5 張
        draw = random.sample(deck, 5 - len(community))
        board = community + draw

        # 評分並累計
        my_best  = best_of_seven(hole,  board)
        opp_best = best_of_seven(opp,    board)
        if   my_best > opp_best: wins += 1
        elif my_best == opp_best: ties += 1

    total = i + 1  # 真實模擬次數
    win_rate  = wins  / total
    tie_rate  = ties  / total
    loss_rate = 1 - win_rate - tie_rate

    return win_rate, tie_rate, loss_rate
        
def _calc_winrate_river(self, my_hole, board):
    """
    回傳 (win_rate, tie_rate)
    - my_hole: 你的兩張手牌，例如 ["HA", "KD"]
    - board  : 5 張公共牌
    """
    # ---------- 1. 準備剩餘牌堆 ----------
    seen = set(my_hole + board)
    remaining = [c for c in self._deck if c not in seen]

    # ---------- 2. 計算自己手牌等級 ----------
    my_best = best_of_seven(my_hole, board)

    # ---------- 3. 枚舉對手兩張手牌 ----------
    win  = 0
    tie  = 0
    total = 0
    for opp in itertools.combinations(remaining, 2):
        opp_best = best_of_seven(list(opp), board)

        # ---------- 4. 比較牌力 ----------
        if my_best > opp_best:        # **贏**
            win += 1
        elif my_best == opp_best:     # **平手**
            tie += 1
        # 否則為輸

        total += 1

    # ---------- 5. 回傳機率 ----------
    return win / total, tie / total

def _detect_draws(self, hole, board):
    """
    回傳 (flush_draw, straight_draw, outs)
    • flush_draw: 距離成同花只差 1 張
    • straight_draw: open-ended 或 gutshot 皆算 True
    • outs: 剩餘可把牌做成抽牌的牌數 (不重複計數，最大 9)
    """
    cards      = hole + board
    board_len  = len(board)
    remaining  = 52 - len(cards)

    # ---------- 1. Flush-Draw ----------
    suits = [c[0] for c in cards]
    flush_draw = False
    outs = 0
    if board_len <= 4:                                     # River 再無新牌就不算 draw
        for s in self._suit:                               # 'C','D','H','S'
            cnt_total  = suits.count(s)
            cnt_board  = [c[0] for c in board].count(s)
            cnt_hero   = cnt_total - cnt_board
            # 四張同花（任一型態）
            if cnt_total == 4 and cnt_hero >= 1:           # Hero 至少持 1 張
                flush_draw = True
                outs = 9
                break
            # Turn 出現 board 4 同花且 Hero 持同色 1 張
            if board_len == 4 and cnt_board == 4 and cnt_hero == 1:
                flush_draw = True
                outs = 9
                break

    # ---------- 2. Straight-Draw ----------
    straight_draw = False
    potential_outs = 0
    # 建立點數的 bitmask (依 self._rank_order "23456789TJQKA")
    bitmask = 0
    for card in cards:
        index = self._rank_order.index(card[1])
        bitmask |= (1 << index)

    # 檢查一般 5 張連續區間（從 '2'~'6' 到 '10'~'A'）
    for low in range(0, 9):  # 範圍 0~8 (共 9 組)
        mask5 = ((1 << 5) - 1) << low  # 產生 5 連的 bitmask
        count = bin(bitmask & mask5).count("1")
        if count == 4:
            missing = mask5 & (~bitmask)
            # 若缺少的牌在區間的邊緣，即 open-ended
            if missing == (1 << low) or missing == (1 << (low + 4)):
                potential_outs = max(potential_outs, 8)
            else:
                potential_outs = max(potential_outs, 4)
            straight_draw = True

    # 檢查輪子 (A,2,3,4,5)，此處 Ace 當作最高牌用 index 12
    wheel_mask = (1 << 12) | ((1 << 4) - 1)  # 包含 Ace (index 12) 和 2~5 (index 0~3)
    count_wheel = bin(bitmask & wheel_mask).count("1")
    if count_wheel == 4:
        potential_outs = max(potential_outs, 8)
        straight_draw = True

    outs = max(outs, potential_outs)

    # ---------- 3. 限制 outs 不超過剩餘牌 ----------
    outs = min(outs, remaining)

    # ---------- 4. 回傳 ----------
    #  (若需 open_ended / gutshot 可一起回)
    print(f"[detect_draws] flush_draw={flush_draw}, straight_draw={straight_draw}, outs={outs}")
    return flush_draw, straight_draw, outs


def _board_texture(self, community):
    suits = [c[0] for c in community]
    ranks = sorted("23456789TJQKA".index(c[1]) for c in community)
    # 同花判斷：兩張同花 = semi、三張同花 = wet
    suit_cnt = max(suits.count(s) for s in set(suits))
    flush_wet = suit_cnt == 3
    flush_semi= suit_cnt == 2
    # 連張判斷：連張差<=1 → 濕；<=2 → 半濕
    is_conn   = ranks[-1] - ranks[0]
    straight_wet  = is_conn <= 2
    straight_semi = is_conn == 3
    # Pair 會讓板面更乾
    has_pair = len(set(ranks)) < 3
    # 綜合
    if flush_wet or straight_wet:
        return "wet"
    if flush_semi or straight_semi:
        return "semi"
    if has_pair:
        return "very_dry"
    return "dry"

def _calc_implied(self, round_state, pot_size):
    """根據 SPR (有效籌碼 / pot) 給一個 0.12–0.40 的隱含賠率係數"""
    # 有效籌碼（所有仍參與玩家 stack 的最小值）
    eff_stack = min(s["stack"] for s in round_state["seats"]
                    if s["state"] == "participating")

    spr = eff_stack / pot_size if pot_size else 0

    # 1) 連續型（滑順）做法
    implied = 0.12 + 0.02 * spr        # 每 +1 SPR 增加 0.02
    implied = max(0.12, min(implied, 0.40))   # 夾在 12%–40%

    # 2) 也可用分段：
    # if   spr < 3:  implied = 0.15
    # elif spr < 8:  implied = 0.22
    # elif spr < 15: implied = 0.28
    # else:          implied = 0.35
    return implied

def _pair_rank(self, hole, board):
    """
    回傳與 Board 配成對子的最大 rank index（0~12）。
    - 若 hole 兩張本身是一對，也算進去。
    - 若完全沒有配對，回傳 -1。
    """
    # 把 board 轉成 rank set 方便查詢
    board_ranks = {c[1] for c in board}
    max_idx = -1               # -1 表示沒有任何配對

    # 1) hole 與 board 配對
    for card in hole:
        if card[1] in board_ranks:
            idx = self._rank_order.index(card[1])
            max_idx = max(max_idx, idx)

    # 2) 考慮 hole 口袋對
    if hole[0][1] == hole[1][1]:
        idx = self._rank_order.index(hole[0][1])
        max_idx = max(max_idx, idx)

    return max_idx             # 固定回傳 int；無配對時 = -1


def _is_top_pair(self, hole, board):
    """判斷是否擊中頂對 (手牌任一張 = board 最高 rank)"""
    if not board: 
        return False
    board_high = max(board, key=lambda c: self._rank_order.index(c[1]))[1]
    print(f"[is_top_pair] board_high={board_high}, hole={hole}")
    return any(c[1] == board_high for c in hole)

def _has_strong_draw(self, hole, board):
    """OESD or 9 outs Flush draw"""
    flush_d, straight_d, outs = self._detect_draws(hole, board)
    return outs >= 8

def _board_four_to_straight(self, board):
    """Flop/Turn/River 是否有四張連號（含輪子 A-2-3-4)"""
    vals = sorted(set(self._rank_order.index(c[1]) for c in board))
    # 把 A=12 再映射成 -1，方便偵測 A-2-3-4
    if 12 in vals:
        vals.append(-1)
        vals.sort()                       # ← 重新排序，才不會打亂遞增序列
    # 任意連續 4 張（不重複）且跨度 = 3
    return any(vals[i+3] - vals[i] == 3 for i in range(len(vals) - 3))


def _board_four_flush(self, board):
    suits = [c[0] for c in board]
    return any(suits.count(s) >= 4 for s in set(suits))

def _board_pairs(self, board):
    cnt = Counter(c[1] for c in board)
    # 恰好一組 pair
    return sum(1 for v in cnt.values() if v == 2)

def _board_trip(self, board):
    cnt = Counter(c[1] for c in board)
    # 恰好一組 pair
    return sum(1 for v in cnt.values() if v == 3)

def _has_flush_blocker(self, hole, board):
    """
    是否擁有同花阻斷牌（以 board 的最大花色為準）
    回傳 True / False
    """
    # 找出公共牌最常見的花色
    suits = [c[0] for c in board]
    major_suit = max(set(suits), key=suits.count)
    # 你手上若持有同花 A 或 K，即視為重大 blocker
    for c in hole:
        if c[0] == major_suit and c[1] in ("A", "K"):
            return True
    return False

def _has_straight_blocker(self, hole, board):
    """
    當 board 四順時，判斷是否持有阻斷張
    （簡化：若你手牌 rank 直接嵌在順子中段，則視為 blocker）
    """
    b_r = sorted({self._rank_order.index(c[1]) for c in board})
    # 找「連續 4 張」的起點
    for i in range(len(b_r) - 3):
        window = b_r[i : i + 4]
        if window[3] - window[0] == 3:          # 0,1,2,3 → 四連
            need = {self._rank_order[(window[0] + j) % 13] for j in range(5)}
            my_ranks = {c[1] for c in hole}
            if my_ranks & need:                 # 有交集 = blocker
                return True
    return False

def _borad_danger(self, hole, board):
    danger = 0
    if self._board_four_to_straight(board): 
        danger += 0.07
    if self._board_four_flush(board): 
        danger += 0.08
    if self._board_pairs(board) > 0: 
        danger += 0.05
    if self._board_trip(board): 
        danger += 0.06
    if self._has_flush_blocker(hole, board): 
        danger -= 0.04
    if self._has_straight_blocker(hole, board): 
        danger -= 0.03

    danger = max(0, danger)

    return danger