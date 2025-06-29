import random
import time
import itertools
from collections import Counter
import torch
from game.players import BasePokerPlayer
from agents.solver import best_of_seven, hand_rank
from agents.deck import Deck

def normalize_hand(hole_cards):
    # 定義牌面大小順序，index 越大代表牌越大
    rank_order = "23456789TJQKA"
    
    # 解析出每張牌的 suit（花色）和 rank（點數）
    suits = [card[0]   for card in hole_cards]    # ['H','H']
    ranks = [card[1]   for card in hole_cards]    # ['K','A']
    
    # 決定哪張是高牌、哪張是低牌
    # 依據 rank_order 的 index 比大小
    if rank_order.index(ranks[0]) > rank_order.index(ranks[1]):
        high, low = ranks[1], ranks[0]
    else:
        high, low = ranks[0], ranks[1]
    
    # 判斷是否同花
    suited_flag = 's' if suits[0] == suits[1] else 'o'
    suited_flag = '' if high == low else suited_flag  # 同 rank 時不標記同花

    key = high + low + suited_flag
    # print(f"Normalized hand: {hole_cards} -> {key}")
    
    return key

class HybridPlayer(BasePokerPlayer):

    def __init__(self):
        self.preflop_table = self.load_preflop_csv("agents/preflop_equity_10000.csv")
        self._rank_order = "23456789TJQKA"
        self._suit = "CDHS"
        self._deck = Deck().full_deck()  
        self.mc_iters = 8000  # 每次模擬的迭代次數
        self.mc_time_limit = 4.0  # 每次模擬的時間限制（秒）
        self.raise_fold = 0

        self.preflop_thresholds = {
            "early":  {"raise_big": 0.65, "raise_small": 0.62, "call": 0.47},
            "middle": {"raise_big": 0.70, "raise_small": 0.65, "call": 0.45},
            "late":   {"raise_big": 0.60, "raise_small": 0.58, "call": 0.42},
        }

    def load_preflop_csv(self, path):
        table = {}
        with open(path) as f:
            next(f)  # 跳過檔頭
            for line in f:
                hand, win_rate, tie_rate = line.strip().split(',')
                table[hand] = float(win_rate)
        return table

    def _clamp (self, value, min_value, max_value):
        """Clamp a value between min_value and max_value."""
        return max(min(value, max_value), min_value)

    def _safe_raise(self, valid_actions, amount, fallback="call", fallback_amt=0):
        """如果可以 raise 就 raise，否則 fallback 動作（預設 call）"""
        for a in valid_actions:
            if a["action"] == "raise" and a["amount"]["min"] != -1 and a["amount"]["max"] != -1:
                min_r = a["amount"]["min"]
                max_r = a["amount"]["max"]
                amt = self._clamp(amount, min_r, max_r)  # clamp
                return "raise", amt
        return fallback, fallback_amt

    def _get_position(self, round_state):
        """
        根據 small_blind_pos 與自己的座位，判定在 Preflop 是 'early' (Big Blind) 還是 'late' (Button/Small Blind)，
        並回傳 (position, thresholds)。
        """
        # 找到自己的座位 index
        seats      = round_state["seats"]            # list of seat dicts
        sb_pos     = round_state["small_blind_pos"]  # small blind 的座位 index

        # 建 map: uuid -> seat_idx
        uuid_to_idx = {p["uuid"]: idx for idx, p in enumerate(seats)}
        my_idx     = uuid_to_idx[self.uuid]
        
        # Heads-Up: small blind 是 button (late)，big blind 在 early
        if my_idx == sb_pos:
            position = "late"
        else:
            position = "early"

        return position  

    def _technical_fold(self, round_state):
        """
        must win if fold in next rounds
        """
        sb       = round_state["small_blind_amount"]
        pos      = self._get_position(round_state)
        my_stack = next(s["stack"] for s in round_state["seats"] if s["uuid"] == self.uuid)
        current_round = round_state["round_count"]
        remaining_rounds = 20 - current_round

        if current_round <= 15: # avoid technical fold in first 15 rounds
            return False

        if remaining_rounds % 2 == 0:
            if my_stack - 1000 > remaining_rounds * (sb * 3) * 0.5:
                print("[Technical Fold] must win if fold in next rounds")
                return True
            else:
                return False
        else:
            if pos == "late":
                if my_stack - 1000 > (remaining_rounds-1) * (sb * 3) * 0.5 + sb:
                    print("[Technical Fold] must win if fold in next rounds")
                    return True
                else:
                    return False
            elif pos == "early":
                if my_stack - 1000 > (remaining_rounds-1) * (sb * 3) * 0.5 + sb * 2:
                    print("[Technical Fold] must win if fold in next rounds")
                    return True
                else:
                    return False

    def _can_fold(self, round_state):
        """
        must win if fold in next rounds
        """
        sb       = round_state["small_blind_amount"]
        pos      = self._get_position(round_state)
        my_stack = next(s["stack"] for s in round_state["seats"] if s["uuid"] == self.uuid)
        current_round = round_state["round_count"]
        remaining_rounds = 20 - current_round
        
        if current_round <= 15:
            return True  # 前 12 輪可以隨意棄牌

        if remaining_rounds % 2 == 0:
            if 1000 - my_stack > remaining_rounds * (sb * 3) * 0.5:
                print("[Cannot Fold] Cannot fold in the next rounds")
                return False
            else:
                return True
        else:
            if pos == "late":
                if 1000 - my_stack > (remaining_rounds-1) * (sb * 3) * 0.5 + sb:
                    print("[Cannot Fold] Cannot fold in the next rounds")
                    return False
                else:
                    return True
            elif pos == "early":
                if 1000 - my_stack > (remaining_rounds-1) * (sb * 3) * 0.5 + sb * 2:
                    print("[Cannot Fold] Cannot fold in the next rounds")
                    return False
                else:
                    return True

        return True

    def decide_preflop(self, valid_actions, hole, round_state):
        if self._technical_fold(round_state): return valid_actions[0]["action"], 0  # technical fold

        pos     = self._get_position(round_state)
        thr     = self.preflop_thresholds[pos]
        winrate = self.preflop_table.get(normalize_hand(hole), 0)
        print(f"[Preflop] Pos: {pos}, Win: {winrate:.2f}, Thr: {thr}")

        # ───────────── 共用變量 ─────────────
        pot      = round_state["pot"]["main"]["amount"]
        call_amt = valid_actions[1]["amount"]
        bb       = round_state["small_blind_amount"] * 2
        min_r, max_r = valid_actions[2]["amount"]["min"], valid_actions[2]["amount"]["max"]
        seats    = round_state["seats"]
        stack_eff = min(p["stack"] for p in seats if p["state"] == "participating")
                
        pot_odds  = call_amt / (pot + call_amt) if call_amt else 0
        margin    = 0.02
        #───────────────────────────────────

        # 0) 強牌：3-bet／shove
        if winrate > thr["raise_big"]:
            # ==== 先判斷有效籌碼深度（stack_eff） ====
            eff_bb = stack_eff / bb
            last_open_amt = max(call_amt, bb)

            # 定義不同 SPR 區間的尺寸策略
            if eff_bb <= 12:
                # 浅籌碼：50% 機率全壓，50% 機率 2.5× open
                if random.random() < 0.5:
                    bet_amt = max_r
                else:
                    bet_amt = int(last_open_amt * 2.5)
            elif eff_bb <= 20:
                # 中等籌碼：70% 用 3× open，30% all-in
                if random.random() < 0.7:
                    bet_amt = int(last_open_amt * 3)
                else:
                    bet_amt = max_r
            elif eff_bb <= 40:
                # 深籌碼：完全 3× open
                bet_amt = int(last_open_amt * 3)
            else:
                # 非常深：2.5–3.5× open 隨機
                factor  = random.uniform(2.5, 3.5)
                bet_amt = int(last_open_amt * factor)

            # clamp 合法範圍
            print(f"[Preflop] Monster hand : eff_bb={eff_bb} 3-bet, bet_amt={bet_amt}")
            return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

        # -------- 0.5) Mixed 3-bet Bluff --------  
        if pos == "late" and 3*bb <= call_amt <= 4*bb:
            if thr["call"] <= winrate < thr["raise_small"]:
                if random.random() < 0.18:                         # 18 % 時間進行 bluff 3-bet
                    factor   = random.uniform(2.7, 3.3)
                    bet_amt  = int(call_amt * factor)
                    print("[Preflop] 3-bet bluff", bet_amt)
                    return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

        
        # 1) 遭遇 limp ⇒ Isolation Raise
        # 條件：對手只有 limp (call_amt == bb) ；自己還能 raise；手牌達到某 winrate
        print(f"[Preflop] call_amt={call_amt}")
        if call_amt == bb and pos == "late":
            # 以 Seat 位置設定 Iso-Raise 閾值（比普通 raise_small 再寬一點）
            print(f"[Preflop] limp call_amt")

            iso_thresh = thr["raise_small"]     # 例：late 0.50 → 0.45
            if winrate >= iso_thresh:
                # Isolation Size：2.5–3.5 BB 隨機
                factor   = random.uniform(2.5, 4.5)
                bet_amt  = int(bb * factor)
                return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)
    
            print("[Preflop] card is not good")
            if winrate < 0.40 and self._can_fold(round_state):
                print("[Preflop] winrate < 0.40")
                return valid_actions[0]["action"], 0
            else:
                return valid_actions[1]["action"], call_amt  # check

        # 2) 依對手加注大小微調 call 閾值，並根據閾值決定 Call / Fold
        print(f"[Preflop] pot_odds={pot_odds:.2f}, stack_eff={stack_eff}")
        # 若面對超過 40% 有效籌碼的全壓／再加注
        if call_amt > 0.4 * stack_eff:
            # 僅在手牌勝率 > 0.64 (大約頂端 10%) 才跟
            if winrate >= 0.64 or not self._can_fold(round_state):
                return valid_actions[1]["action"], call_amt
            else:
                return valid_actions[0]["action"], 0
        
        if call_amt >= 5 * bb or call_amt > 0.25 * stack_eff:

            if winrate >= pot_odds + margin:
                print("[Preflop] Hero-call vs Shove")
                return valid_actions[1]["action"], call_amt  # call

            call_thresh = max(thr["raise_big"], pot_odds + margin)
            if winrate >= call_thresh or not self._can_fold(round_state):
                print(f"[Preflop] Call (winrate={winrate:.2f}, call_thresh={call_thresh:.2f})")
                factor = random.uniform(2.5, 3.0)
                bet_amt = int(call_amt * factor)
                return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)
            else:
                print(f"[Preflop] Fold (winrate={winrate:.2f}, call_thresh={call_thresh:.2f})")
                self.raise_fold += 1
                return valid_actions[0]["action"], 0
            
        
        if call_amt >= 3 * bb:
            call_thresh = max(thr["call"] + 0.08    , pot_odds + margin)
        else:
            call_thresh = max(thr["call"], pot_odds + margin)

        if winrate >= call_thresh or not self._can_fold(round_state):
            print(f"[Preflop] Call (winrate={winrate:.2f}, call_thresh={call_thresh:.2f})")
            return valid_actions[1]["action"], call_amt
        else:
            print(f"[Preflop] Fold (winrate={winrate:.2f}, call_thresh={call_thresh:.2f})")
            self.raise_fold += 1
            return valid_actions[0]["action"], 0


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
        flush_wet = suit_cnt >= 3
        flush_semi= suit_cnt >= 2
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

    def _board_high(board):
        """回傳 board 上最高牌的 rank index (0~12)"""
        if not board: 
            return -1
        return max(self._rank_order.index(c[1]) for c in board)

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
        print(f"[has_strong_draw] outs={outs}")
        return outs >= 8

    def _board_four_to_straight(self, board):
        """Flop/Turn/River 是否有四張連號 draw（包括 open‐ended 與 gutshot）"""
        # 用 bitmask 表示各點數在 board 上是否出現
        bitmask = 0
        for c in board:
            idx = self._rank_order.index(c[1:])
            bitmask |= (1 << idx)
        # Ace 作為 1
        if bitmask & (1 << 12):
            bitmask |= 1  # 把 bit 0 (rank '2') 也標成已出現，等於 A-2-3-4

        # 掃所有可能的 5 張連號區間
        for low in range(0, 9):  # 0..8 共 9 種 5 張組合
            mask5 = ((1 << 5) - 1) << low
            if bin(bitmask & mask5).count("1") == 4:
                return True
        return False

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
            danger += 0.05 * self._board_pairs(board)
        if self._board_trip(board): 
            danger += 0.06
        if self._has_flush_blocker(hole, board): 
            danger -= 0.04
        if self._has_straight_blocker(hole, board): 
            danger -= 0.03

        danger = max(0, danger)

        return danger

    def decide_flop(self, valid_actions, hole_card, round_state):
        community = round_state["community_card"]
        win_mc, _, _ = self.estimate_winrate_mc(hole_card, community, iterations=4000)
        texture = self._board_texture(community)
        best = best_of_seven(hole_card, community)
        rank = best[0]

        # 1) 計算 pot size & call amount
        pot_size   = round_state["pot"]["main"]["amount"]
        min_r = valid_actions[2]["amount"]["min"]
        max_r = valid_actions[2]["amount"]["max"]
        call_amt   = valid_actions[1]["amount"]
        print(f"[Flop] pot_size={pot_size}, call_amt={call_amt}, texture={texture}")
        pot_odds   = call_amt / (pot_size + call_amt)
        stack_eff = min(s["stack"] for s in round_state["seats"] if s["state"] == "participating")   
        spr = stack_eff / pot_size if pot_size > 0 else 0
        margin = 0.05
        implied = self._calc_implied(round_state, pot_size)
        print(f"[Flop] win_mc={win_mc:.2f}, pot_odds={pot_odds:.2f}, rank={rank}, spr={spr:.2f}")    

        if call_amt == 0: # active
            print("[decide_flop] call_amt == 0, active player")

            if self._is_top_pair(hole_card, community) and rank >= 2:
                print("[decide_flop] 頂對強制下注")
                pct = 0.50 if texture in ("wet", "semi") else 0.40
                bet_amt = int(pot_size * pct)
                return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

            elif rank >= 5:
                pct = 0.7 if texture == "wet" else 0.55
                bet_amt = int(pot_size * pct)
                return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

            elif rank in (3, 4) and texture in ("dry", "semi"):
                pct      = 0.45 if texture in ("dry", "semi") else 0.55
                bet = int(pot_size * pct)
                return self._safe_raise(valid_actions, bet)

            # 4) 強抽但尚未成手 (rank < 2)             －－半閃
            elif rank < 2 and self._has_strong_draw(hole_card, community) and win_mc >= pot_odds + margin:
                if spr > 12 and win_mc < 0.6:
                    # SPR 太高、勝率不夠高 → 不下注
                    return valid_actions[1]["action"], call_amt
                pct      = 0.55 if texture == 'wet' else 0.75
                bet_amt  = int(pot_size * pct)
                return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

            elif rank == 1 and texture != "wet":
                print(f"[Flop] texture={texture}, little bet")
                
                if self._is_top_pair(hole_card, community) and win_mc > 0.7:  # 高對
                    pct = 0.6
                    bet_amt  = int(pot_size * pct)
                    return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)
                else:  # 低對 
                    return valid_actions[1]["action"], call_amt  # free check

            else:
                print(f"[decide_flop] free check")
                return valid_actions[1]["action"], call_amt  # free check

        else: # passive
            if call_amt >= 0.4 * pot_size or call_amt > 0.25 * stack_eff: # heavy bet
                print(f"[decide_flop] call_amt={call_amt}, heavy bet")

                if spr < 3 and self._has_strong_draw(hole_card, community):
                    print("[Flop] SPR<3 且有强抽 → all in semi bluff")
                    return valid_actions[2]["action"], max_r  # all-in

                if rank == 8: # shove
                    bet_amt = max_r
                    return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

                if rank == 7:
                    bet_amt = max_r if spr <= 1.5 else int(pot_size * random.uniform(1.1,1.4))
                    return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

                if rank == 6:
                    pct = 1.1 if texture in ("wet","semi") else 0.8
                    bet_amt = int(pot_size * pct)
                    return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

                if rank == 5:
                    if texture in ("wet", "semi"): bet_amt = int(pot_size * random.uniform(0.5, 0.7))

                    else: bet_amt = int(pot_size * random.uniform(0.6, 0.8))

                    return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

                if rank in (3,4) or (rank == 2 and self._is_top_pair(hole_card, community)):
                    if win_mc < pot_odds - margin and self._can_fold(round_state):
                        return 'fold', 0
                    # Thin value：0.45–0.55 pot
                    bet_amt = int(pot_size * random.uniform(0.45,0.55))
                    return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

                # 5) 其他牌
                if rank <= 1 and call_amt > 0.5 * stack_eff:
                    print("[Flop] Weak hand vs all-in → fold")
                    return valid_actions[0]["action"], 0
                if win_mc >= pot_odds + margin:
                    return 'call', call_amt
                else:
                    return 'fold', 0
            
            else:  # light bet
                print(f"[decide_flop] call_amt={call_amt}, light bet")
                if rank >= 6:
                    bet_amt = int(pot_size * random.uniform(0.7, 0.9))
                    return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt) 
                elif rank == 5:
                    if texture in ("dry", "very_dry"):
                        bet_amt = int(pot_size * random.uniform(0.4, 0.6))
                        return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)  
                    else:  # wet / semi
                        return valid_actions[1]["action"], call_amt  # call
                elif rank in (3, 4):
                    if texture in ("dry", "very_dry"):
                        bet_amt = int(pot_size * random.uniform(0.3, 0.5))
                        return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)
                    else:  # wet / semi
                        return valid_actions[1]["action"], call_amt  # call
                elif rank in (1, 2):
                    # rank 1, 2 都是弱牌
                    if win_mc < pot_odds + margin and self._can_fold(round_state):
                        self.raise_fold += 1
                        return valid_actions[0]["action"], 0

                    if (not self._is_top_pair(hole_card, community) or self._has_strong_draw(hole_card, community)) or spr < 1.5:  # 高對
                        return valid_actions[1]["action"], call_amt

                    # if winrate < 0.65 and not self._has_strong_draw(hole_card, community):
                    #     return valid_actions[0]["action"], 0
                    
                    pct = 0.35
                    bet_amt = int(pot_size * pct)
                    return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

                else: # rank 0
                    flush_d, straight_d, outs = self._detect_draws(hole_card, community)
                    if outs >= 8:  # 強抽
                        factor = 0.6 if spr < 4 else 0.45
                        bet_amt = int(pot_size * factor)
                        return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)  # 強抽，下注
                    
                    if outs >= 4 and win_mc >= pot_odds + margin:
                        return valid_actions[1]["action"], call_amt  # call

                    if self._can_fold(round_state):
                        self.raise_fold += 1
                        return valid_actions[0]["action"], 0
                    else:
                        return valid_actions[1]["action"], call_amt
  
    def decide_turn(self, valid_actions, hole_card, round_state):
        street = round_state["street"]
        community = round_state["community_card"]
        
        best = best_of_seven(hole_card, community)
        rank = best[0]
        win_mc, tie_mc, _ = self.estimate_winrate_mc(hole_card, community, iterations = 3000)
        texture = self._board_texture(community)
        print(f"[Turn] win_mc={win_mc:.2f}, rank={rank}, texture={texture}")  

        # 2) 讀池＆計算 Pot Odds
        min_r, max_r = valid_actions[2]["amount"]["min"], valid_actions[2]["amount"]["max"]
        pot_size = round_state["pot"]["main"]["amount"]
        call_amt = valid_actions[1]["amount"]
        pot_odds = call_amt / (pot_size + call_amt) if call_amt > 0 else 0
        stack_eff = min(s["stack"] for s in round_state["seats"] if s["state"] == "participating")
        spr = stack_eff / pot_size if pot_size > 0 else 0
        margin   = 0.03  # 安全邊際

        # 3) Danger analysis
        danger = self._borad_danger(hole_card, community)
        adj_win = win_mc - danger
        print(f"[Turn] danger={danger:.2f}, adj_win={adj_win:.2f}")

        if call_amt > 0:
            print(f"[Turn] passive player")
            if call_amt >= 0.7 * pot_size or call_amt > 0.40 * stack_eff: # heavy bet
                print(f"[Turn] HEAVY BET: call_amt={call_amt}, pot_size={pot_size}, rank={rank}")
                if rank in (8, 7):  # shove
                    print(f"[Turn] rank={rank}, heavy bet, shove")
                    bet_amt = max_r
                    if spr > 2:
                        pct = random.uniform(1.2, 1.6)
                        bet_amt = int(pot_size * pct)
                    return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

                if rank == 6:
                    pct = random.uniform(1.0, 1.2)
                    bet_amt = int(pot_size * pct)
                    return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

                elif rank == 5:
                    if texture in ("dry, very_dry"):
                        pct = random.uniform(0.6, 0.8)
                        bet_amt = int(pot_size * pct)
                        return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)
                    else:  # wet / semi
                        return valid_actions[1]["action"], call_amt  # call
                
                elif rank in (3, 4):
                    print(f"[decide_turn] texture={texture} pot_odds={pot_odds:.2f}, adj_win={adj_win:.2f}")
                    if adj_win < pot_odds + margin and self._can_fold(round_state):
                        self.raise_fold += 1
                        return valid_actions[0]["action"], 0
                    elif texture in ("wet", "semi"):
                        return valid_actions[1]["action"], call_amt  # call
                    else:
                        bet_amt = int(pot_size * 0.2 + call_amt)
                        return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

                elif rank in (1, 2):
                    print(f"[Turn] texture={texture}, adj_win={adj_win:.2f}, pot_odds={pot_odds:.2f}")
                    if adj_win < pot_odds + margin and self._can_fold(round_state):
                        self.raise_fold += 1
                        return valid_actions[0]["action"], 0

                    elif texture in ("wet", "semi") or self._has_strong_draw(hole_card, community):
                        r = random.random()
                        if r < 0.75:
                            bet_amt = int(pot_size * 2.5)
                            return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)
                        else:
                            
                            return valid_actions[1]["action"], call_amt
                    else:
                        return valid_actions[1]["action"], call_amt  # call
                
                else:
                    if self._has_strong_draw(hole_card, community):

                        if pot_odds < (outs / 46):
                            return valid_actions[1]["action"], call_amt  # call

                        bet_amt = int(pot_size * 0.5)
                        return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

                    if self._can_fold(round_state):
                        self.raise_fold += 1
                        return valid_actions[0]["action"], 0
                    else:
                        return valid_actions[1]["action"], call_amt
                        

            else: # light bet
                print(f"[decide_turn] call_amt={call_amt}, light bet")

                if rank >= 6:
                    pct = random.uniform(0.7, 0.8)
                    bet_amt = int(pot_size * pct)
                    return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

                elif rank == 5:
                    if texture in ("dry, very_dry"):
                        pct = random.uniform(0.5, 0.6)
                        bet_amt = int(pot_size * pct)
                        return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)
                    else:  # wet / semi
                        return valid_actions[1]["action"], call_amt  # call
                
                elif rank in (3, 4):
                    if texture in ("wet", "semi") and adj_win < 0.85 and self._can_fold(round_state):
                        self.raise_fold += 1
                        return valid_actions[0]["action"], 0
                    else:
                        pct = random.uniform(0.3, 0.5)
                        bet_amt = int(pot_size * pct)
                        return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

                elif rank in (1, 2):
                    print(f"[Turn] texture={texture}, adj_win={adj_win:.2f}, pot_odds={pot_odds:.2f}")
                    if adj_win < pot_odds + margin and self._can_fold(round_state):
                        self.raise_fold += 1
                        return valid_actions[0]["action"], 0
                    else:
                        return valid_actions[1]["action"], call_amt

                else:  # rank == 0
                    print(f"[Turn] AK? {any(c[1] in ('A', 'K') for c in hole_card)}")
                    if spr <= 4 and self._has_strong_draw(hole_card, community) and any(c[1] in ('A', 'K') for c in hole_card): # has A or K
                        print(f"[decide_turn] 強抽")
                        pct = random.uniform(0.5, 0.6)
                        bet_amt = int(pot_size * pct)
                        return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

                    elif self._can_fold(round_state) and adj_win < 0.5 and adj_win < pot_odds + margin:
                        self.raise_fold += 1
                        return valid_actions[0]["action"], 0
                    else:
                        return valid_actions[1]["action"], call_amt

        else:  # call_amt == 0, active player
            print(f"[decide_turn] active player, pot_size={pot_size}, rank={rank}")
            if rank >= 6: # value-bet
                print("[decide_turn] good hand, value-bet")
                pct = 0.75 if texture == "wet" else 0.85
                bet_amt = int(pot_size * pct)
                return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

            elif rank >= 4:
                print("[decide_turn] 中等牌 thin value-bet")
                pct = 0.5 if texture in ("wet", "semi") else 0.4
                bet_amt = int(pot_size * pct)
                return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)
            
            elif rank >= 1 and self._has_strong_draw(hole_card, community) and spr > 4:
                print("[decide_turn] 強抽")
                pct = 0.6 if texture == 'wet' else 0.3
                bet_amt = int(pot_size * pct)
                return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

            elif rank < 2 and self._is_top_pair(hole_card, community):
                print("[decide_turn] 邊緣頂對，block bet")
                pct = 0.25
                bet_amt = int(pot_size * pct)
                return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

            elif rank in (1, 2):
                if adj_win > 0.78:
                    bet_amt = int(pot_size * 0.70)
                    print(f"[decide_turn] rank={rank}, adj_win={adj_win:.2f}>0.75, bet_amt={bet_amt}")
                    return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)
                else:
                    return valid_actions[1]["action"], call_amt  # free check

            else:
                print(f"[decide_turn] free check")
                return valid_actions[1]["action"], call_amt  # free check

    def decide_river(self, valid_actions, hole_card, round_state):
        street = round_state["street"]
        community = round_state["community_card"]
        # 1) 直接計算精準勝率
        best   = best_of_seven(hole_card, community)
        rank   = best[0]        # 0: high-card … 8: straight-flush
        win_r, tie_r = self._calc_winrate_river(hole_card, community)
        texture = self._board_texture(community)
        print(f"[River] win_mc={win_r:.2f}, rank={rank}")

        # 3) 讀池＆計算 Pot Odds
        min_r, max_r   = valid_actions[2]["amount"]["min"], valid_actions[2]["amount"]["max"]
        pot_size = round_state["pot"]["main"]["amount"]
        call_amt = valid_actions[1]["amount"]
        pot_odds = call_amt / (pot_size + call_amt) if call_amt > 0 else 0
        margin   = 0.03  # 安全邊際
        stack_eff = min(s["stack"] for s in round_state["seats"] if s["state"] == "participating")
        spr = stack_eff / pot_size if pot_size > 0 else 0
        print(f"[River] pot_size={pot_size}, call_amt={call_amt}, pot_odds={pot_odds:.2f}")

        # —— River bluff-catch 規則 ——
        # ---------- 危險牌面 + Blocker Bonus ----------
        danger = self._borad_danger(hole_card, community)
        adj_win = win_r - danger
        print(f"[River] adj_win={adj_win:.2f}, danger={danger:.2f}")

        if call_amt > 0:
            print(f"[River] passive player")
            if call_amt >= 0.4 * pot_size:
                print(f"[River] call_amt={call_amt}, pot_size={pot_size} heavy bet")
                if rank == 8: # shove
                    print(f"[River] rank={rank}, shove")
                    bet_amt = max_r
                    return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

                elif rank == 7:
                    bet_amt = max_r if spr <= 1.5 else int(pot_size * random.uniform(1.0, 1.5))
                    return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

                elif rank == 6:
                    if texture == "wet":
                        bet_amt = int(1.2 * pot_size)
                    if texture == "semi":
                        bet_amt = int(1.0 * pot_size)
                    else:  # dry / very_dry
                        bet_amt = int(0.8 * pot_size)

                    bet_amt = self._clamp(bet_amt, min_r, max_r)

                elif rank == 5:
                    if texture in ("wet", "semi"):
                        return valid_actions[1]["action"], call_amt  # call
                    else:
                        bet_amt = int(pot_size * random.uniform(0.6, 0.8))
                        return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

                elif rank in (3, 4):
                    if adj_win < pot_odds - margin and self._can_fold(round_state):
                        print(f"win rate too low, fold")
                        self.raise_fold += 1
                        return valid_actions[0]["action"], 0 # fold

                    if texture in ("wet", "semi"):
                        bet_amt = int(pot_size * random.uniform(0.20, 0.25))
                    else:
                        bet_amt = int(pot_size * random.uniform(0.30, 0.35))
                    return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)
                
                elif rank in (0, 1, 2):
                    print(f"[River] rank={rank}, adj_win={adj_win:.2f}, pot_odds={pot_odds:.2f}, texture={texture}")
                    if (adj_win < 0.75 or adj_win < pot_odds - margin) and self._can_fold(round_state):
                        print(f"win rate too low, fold")
                        self.raise_fold += 1
                        return valid_actions[0]["action"], 0
                    else:
                        return valid_actions[1]["action"], call_amt  # call
            
            else:  # light bet
                print(f"[River] call_amt={call_amt}, pot_size={pot_size} light bet")

                # ── 1. Monster / 強價值 ───────────────────────────────
                if   rank == 8:                 # Straight-flush / Quads
                    print("[River-LB] Rank 8 → pot-raise")
                    bet_amt = int(pot_size * 1.2)
                    return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

                elif rank == 7:                 # Full House
                    bet_amt = int(pot_size * 1.1)
                    return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

                elif rank == 6:                 # Nut Flush / Nut Straight
                    bet_amt = int(pot_size * 0.8)
                    return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

                # ── 2. 中強 (Trips / 強兩對) ──────────────────────────
                elif rank == 5:
                    # light bet → 直接加到 0.6-0.7 pot，防止對手看廉價攤牌
                    bet_amt = int(pot_size * random.uniform(0.60, 0.70))
                    return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

                elif rank in (3, 4):            # Two-pair / Over-pair
                    # 計算 bluff-catch EV
                    if adj_win < pot_odds - margin and self._can_fold(round_state):
                        print("[River-LB] Over-pair / Two-pair but EV<0 → fold")
                        self.raise_fold += 1
                        return valid_actions[0]["action"], 0
                    else:
                        # 半 pot  thin-value 或 block-bet 回擊
                        if texture in ("wet", "semi"):
                            bet_amt = int(pot_size * random.uniform(0.35, 0.45))
                            return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)
                        else:
                            # 乾板直接 call，避免被輕鬆 3-bet
                            return valid_actions[1]["action"], call_amt

                # ── 3. 弱牌 / 單對以下 ──────────────────────────────
                elif rank in (1, 2):                           # rank 0-2
                    print(f"[River-LB] rank={rank}, adj={adj_win:.2f}, pot_odds={pot_odds:.2f}")
                    
                    if adj_win > 0.5 and pot_odds > 0.2:
                        # 只要有超過 50% 的勝率，且對手下注不深，就 thin value bet
                        pct = 0.30
                        bet_amt = int(pot_size * pct)
                        print(f"[River-LB] thin value bet {bet_amt}")
                        return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)
                    
                    if adj_win >= pot_odds + margin and adj_win >= 0.4: # 
                        # +EV 接注 → bluff-catch
                        return valid_actions[1]["action"], call_amt
                    if self._can_fold(round_state):
                        self.raise_fold += 1
                        return valid_actions[0]["action"], 0
                    else:
                        return valid_actions[1]["action"], call_amt
                
                else:  # rank == 0
                    print(f"[River-LB] rank={rank}, adj={adj_win:.2f}, pot={pot_odds:.2f}")
                    if self._can_fold(round_state):
                        self.raise_fold += 1
                        return valid_actions[0]["action"], 0
                    else:
                        return valid_actions[1]["action"], call_amt


        else:
            print(f"[River] Active: rank={rank}, adj_win={adj_win:.2f}, pot_odds={pot_odds:.2f}, texture={texture}")          

            # ---------- Rank Tier 重新分層 ----------
            # 8  = STRAIGHT_FLUSH / QUADS
            # 7  = FULL_HOUSE
            # 6  = FLUSH / STRAIGHT
            # 5  = TRIPS
            # 4  = TWO_PAIR
            # 3  = OVER‐PAIR (口袋對大於 board 高張)
            # 2  = ONE_PAIR (頂對/中對)
            # 0-1= no-made / busted draw

            #   強牌 Tier-A : rank 8
            #   強牌 Tier-B : rank 7
            #   強牌 Tier-C : rank 6
            #   中強       : rank 5
            #   中檔       : rank 3-4
            #   弱牌       : rank ≤2
            # ------------------------------------------------

            # === A. 超強牌（rank 8）→ 大注 / Over-bet ===
            if rank == 8:
                factor = random.uniform(0.9, 2.2 if texture in ("semi", "wet") else 1.4)
                bet_amt = int(pot_size * factor)
                return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

            # === B. Very Strong（rank 7）====
            elif rank == 7:
                factor = random.uniform(0.75, 1.2 if texture in ("semi", "wet") else 1.0)
                bet_amt = int(pot_size * factor)
                return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

            # === C. Strong Value（rank 6）====
            elif rank == 6:
                factor = random.uniform(0.55, 0.85)
                bet_amt = int(pot_size * factor)
                return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

            # === D. 中強（Trips，rank 5）====
            elif rank == 5:
                factor = random.uniform(0.40, 0.60)
                bet_amt = int(pot_size * factor)
                return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

            # === E. 中檔（Two-Pair / Over-Pair，rank 3-4）====
            elif rank in (3, 4):
                danger_flush = self._board_four_flush(community) and not self._has_flush_blocker(hole_card, community)
                danger_straight = self._board_four_to_straight(community) and not self._has_straight_blocker(hole_card, community)
                danger_pairs = ( self._board_pairs(community) > 0 or self._board_trip(community) ) and not self._is_top_pair(hole_card, community)
                if danger_straight or danger_flush or danger_pairs:
                    return valid_actions[1]["action"], call_amt  # 安全 check 

                if danger > 0:                     # 濕板 → 偏小 block
                    factor = random.uniform(0.25, 0.35)
                else:
                    factor = random.uniform(0.35, 0.55)  # 乾板 → 偏大 thin value
                bet_amt = int(pot_size * factor)
                return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

            # === F. 頂對 / 中對（rank 2）或更差 ===
            #     - 若對手下注，進入 bluff-catch 判斷
            #     - 否則可嘗試 20-30% pot 輕偷或 simply check
            print(f"[River] rank={rank}, F.")
            

            if danger > 0:
                roll = random.random()
                has_blocker = self._has_flush_blocker(hole_card, community) or self._has_straight_blocker(hole_card, community)
                print(f"[River] danger={danger:.2f}, has_blocker={has_blocker}, roll={roll:.2f}")
                if texture in ("dry", "very_dry") and roll < 0.70:
                    factor = random.uniform(0.25, 0.35)          # 更能製造 fold equity

                elif texture == "semi" and has_blocker and roll < 0.50:       # semi 降低頻率
                    factor = random.uniform(0.20, 0.25)

                elif texture == "wet" and has_blocker and roll < 0.30:
                    factor = random.uniform(0.20, 0.22)

                else:    # 沒命中偷注條件 → check
                    print(f"[River] skip bluff: no factor assigned")
                    return valid_actions[1]["action"], call_amt

                bet_amt = int(pot_size * factor)
                print(f"[River] DEBUG bet_amt={bet_amt}")
                return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)

            # danger = 0
            if  adj_win >= pot_odds + margin + 0.1 and adj_win >= 0.8:
                print(f"[River] thin value bet")
                factor = random.uniform(0.15, 0.20)
                bet_amt = int(pot_size * factor)
                return self._safe_raise(valid_actions, bet_amt, fallback_amt=call_amt)
            
            print(f"[River] check for safe")
            return valid_actions[1]["action"], call_amt


    def declare_action(self, valid_actions, hole_card, round_state):
        street = round_state["street"]
        community = round_state["community_card"]

        # Preflop
        if street == "preflop":
            return self.decide_preflop(valid_actions, hole_card, round_state)
            
        # Post-flop (flop)
        elif street == "flop":
            return self.decide_flop(valid_actions, hole_card, round_state)
                
        # Turn/River
        elif street == "turn":  # turn or river
            return self.decide_turn(valid_actions, hole_card, round_state)

        elif street == "river":
            return self.decide_river(valid_actions, hole_card, round_state)

    # 其餘 callback 留空或做 logging
    def receive_game_start_message(self, game_info): 
        # print(f"[HybridPlayer] Game started with info: {game_info}")
        pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state):
        # print(f"[HybridPlayer] Street started: {street}, Round state: {round_state}")
        pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

def setup_ai():
    return HybridPlayer()
