import random
import time
import itertools
import traceback
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
            "early":  {"raise_big": 0.68, "raise_small": 0.62, "call": 0.50},
            "middle": {"raise_big": 0.70, "raise_small": 0.65, "call": 0.45},
            "late":   {"raise_big": 0.60, "raise_small": 0.58, "call": 0.43},
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

    def get_position(self, seat_index, button, num_players, small_blind_pos):
        # heads-up 特例
        if num_players == 2:
            # 小盲先出手 → early；大盲後出手 → late
            return "early" if seat_index == small_blind_pos else "late"

        # 多人桌（>2）時，維持原本三區分法
        rel = (seat_index - button - 1) % num_players
        third = num_players // 3
        if rel <= third - 1:
            return "early"
        elif rel <= 2 * third - 1:
            return "middle"
        else:
            return "late"

    def _technical_fold(self, valid_actions, round_state):
        """
        must win if fold in next rounds
        """
        sb = round_state["small_blind_amount"]
        my_stack = next(s["stack"] for s in round_state["seats"] if s["name"] == "myBot")
        opp_stack = next(s["stack"] for s in round_state["seats"] if s["state"] == "participating" and s["name"] != "myBot")
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
        my_stack = next(s["stack"] for s in round_state["seats"] if s["name"] == "myBot")
        current = round_state["round_count"]
        
        if current == 20 and my_stack < sb * 2 + 1000:
            print("[Cannot Fold] Cannot fold in the last round")
            return False
        return True

    def decide_preflop(self, valid_actions, hole, round_state):
        if self._technical_fold(valid_actions, round_state):
            return valid_actions[0]["action"], 0  # technical fold

        seats   = round_state["seats"]
        num_p   = len(seats)
        button  = round_state["dealer_btn"]
        sb_pos  = round_state["small_blind_pos"]
        seat_idx= next(i for i,s in enumerate(seats) if s["name"]=="myBot")
        pos     = self.get_position(seat_idx, button, num_p, sb_pos)
        thr     = self.preflop_thresholds[pos]
        print(f"[Preflop] Position: {pos}, Thresholds: {thr}")

        winrate = self.preflop_table.get(normalize_hand(hole), 0)

        # ───────────── 共用變量 ─────────────
        pot      = round_state["pot"]["main"]["amount"]
        call_amt = valid_actions[1]["amount"]
        bb       = round_state["small_blind_amount"] * 2
        min_r, max_r = valid_actions[2]["amount"].values()
        stack_eff = min(p["stack"] for p in seats if p["state"] == "participating")
                
        pot_odds  = call_amt / (pot + call_amt) if call_amt else 0
        margin    = 0.02

        #───────────────────────────────────

        # =========================================================
        # 0) 強牌：3-bet／shove
        # =========================================================
        if winrate > thr["raise_big"]:
            # ==== 先判斷有效籌碼深度（stack_eff） ====
            
            bb_units  = stack_eff // bb                      # 以 BB 為單位的深度

            # ==== 依深度設定「下注選項」及對應權重 ====
            # format: (下注類型, 權重)
            if bb_units <= 20:         # 淺籌碼 → 以 shove 為主
                options = [("shove", 0.7), ("big", 0.3)]
            elif bb_units <= 40:       # 中等籌碼 → 三種尺寸混合
                options = [("shove", 0.4), ("big", 0.4), ("mid", 0.2)]
            else:                      # 深籌碼 → 以 3-bet 為主、偶爾 all-in
                options = [("shove", 0.2), ("big", 0.5), ("mid", 0.3)]

            # ==== 根據權重抽籤 ====
            choice = random.choices( [opt[0] for opt in options], 
                    weights=[opt[1] for opt in options], k=1)[0]

            # ==== 計算 bet_amt ====
            if choice == "shove":
                bet_amt = max_r                                # all-in
            elif choice == "big":
                factor  = random.uniform(5.0, 6.0)             # 5–6 BB
                bet_amt = max(min_r, min(int(bb * factor), max_r))
            else:  # "mid"
                factor  = random.uniform(3.5, 4.5)             # 3.5–4.5 BB
                bet_amt = max(min_r, min(int(bb * factor), max_r))

            if bet_amt > 0.45 * stack_eff or bet_amt >= 5 * bb:
                bet_amt = max_r # 強制 all-in   

            print(f"[Preflop] Monster hand → {choice} 3-bet, bet_amt={bet_amt}")
            return valid_actions[2]["action"], bet_amt
        
        # =========================================================
        # 1) 遭遇 limp ⇒ Isolation Raise
        # =========================================================
        # 條件：對手只有 limp (call_amt == bb) ；自己還能 raise；手牌達到某 winrate
        print(f"[Preflop] call_amt={call_amt}, winrate={winrate:.2f}, position={pos}")
        if call_amt == bb and pos == "late":
            # 以 Seat 位置設定 Iso-Raise 閾值（比普通 raise_small 再寬一點）
            print(f"[Preflop] limp call_amt={call_amt}, winrate={winrate:.2f}, position={pos}")

            iso_thresh = thr["raise_small"]     # 例：late 0.50 → 0.45
            if winrate >= iso_thresh:
                # Isolation Size：2.5–3.5 BB 隨機
                factor   = random.uniform(2.5, 4.5)
                desired  = int(bb * factor)
                bet_amt  = max(min_r, min(desired, max_r))
                return valid_actions[2]["action"], bet_amt
    
            else:    
                print("[Preflop] card is not good")
                if winrate < 0.35 and self._can_fold(round_state):
                    print("[Preflop] winrate < 0.35")
                    return valid_actions[0]["action"], 0
                else:
                    return valid_actions[1]["action"], call_amt  # check

        # =========================================================
        # 2) 依對手加注大小微調 call 閾值，並根據閾值決定 Call / Fold
        # =========================================================
        print(f"[Preflop] pot_odds={pot_odds:.2f}, stack_eff={stack_eff}, call_amt={call_amt}")
        
        if call_amt >= 6 * bb or call_amt > 0.3 * stack_eff:
            call_thresh = max(thr["raise_small"], pot_odds + margin)
        elif call_amt >= 3 * bb:
            call_thresh = max(thr["call"] + 0.10, pot_odds + margin)
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
        print(f"[detect_draws] hole={hole}, board={board}")
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

    # Support functions
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
        print(f"[has_strong_draw] outs={outs}")
        return outs >= 8

    def _board_four_to_straight(self, board):
        """Flop/Turn/River 是否有四張連號"""
        vals = sorted(set(self._rank_order.index(c[1]) for c in board))
        # 考慮 Wheel
        if 12 in vals: vals.append(-1)
        return any(vals[i+3]-vals[i]==3 for i in range(len(vals)-3))

    def _board_four_flush(self, board):
        suits = [c[0] for c in board]
        return any(suits.count(s) >= 4 for s in set(suits))

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
        stack_eff = min(s["stack"] for s in round_state["seats"]
                        if s["state"] == "participating")   
        spr = stack_eff / pot_size if pot_size > 0 else 0
        margin = 0.03
        implied = self._calc_implied(round_state, pot_size)
        print(f"[Flop] win_mc={win_mc:.2f}, pot_odds={pot_odds:.2f}, rank={rank}, spr={spr:.2f}")    

        if call_amt == 0: # active
            print("[decide_flop] call_amt == 0, active player")
            if self._is_top_pair(hole_card, community) and rank >= 2:
                print("[decide_flop] 頂對強制下注")
                pct = 0.50 if texture in ("wet", "semi") else 0.40
                bet_amt = int(pot_size * pct)
                bet_amt = self._clamp(bet_amt, min_r, max_r)
                return valid_actions[2]["action"], bet_amt

            elif rank >= 5:
                pct = 0.7 if texture == "wet" else 0.55
                bet_amt = int(pot_size * pct)
                bet_amt = self._clamp(bet_amt, min_r, max_r)
                return valid_actions[2]["action"], bet_amt

            elif rank in (3, 4) and texture in ("dry", "semi"):
                pct      = 0.45 if texture in ("dry", "semi") else 0.55
                bet = self._clamp(int(pot_size * pct), min_r, max_r)
                return valid_actions[2]["action"], bet

            # 4) 強抽但尚未成手 (rank < 2)             －－半閃
            elif rank < 2 and self._has_strong_draw(hole_card, community):
                pct      = 0.70 if texture == 'wet' else 0.60
                bet_amt  = self._clamp(int(pot_size * pct), min_r, max_r)
                return valid_actions[2]["action"], bet_amt

            elif rank == 1 and texture != "wet":
                pair_rank = self._pair_rank(hole_card, community)
                print(f"[decide_flop] texture={texture}, pair_rank={pair_rank}, little bet")
                if self._is_top_pair(hole_card, community) and win_mc > 0.7:  # 高對
                    pct = 0.6
                    bet_amt  = self._clamp(int(pot_size * pct), min_r, max_r)
                    return valid_actions[2]["action"], bet_amt
                else:  # 低對 
                    return valid_actions[1]["action"], call_amt  # free check

            else:
                print(f"[decide_flop] free check")
                return valid_actions[1]["action"], call_amt  # free check

        else: # passive
            if call_amt >= 0.4 * pot_size or call_amt > 0.25 * stack_eff: # heavy bet
                print(f"[decide_flop] call_amt={call_amt}, heavy bet")
                if rank == 8: # shove
                    bet_amt = max_r
                    return valid_actions[2]["action"], bet_amt

                if rank == 7:
                    bet_amt = max_r if spr <= 1.5 else int(pot_size * random.uniform(1.1,1.4))
                    return valid_actions[2]["action"], bet_amt

                if rank == 6:
                    pct = 1.1 if texture in ("wet","semi") else 0.8
                    bet_amt = self._clamp(int(pot_size * pct), min_r, max_r)
                    return 'raise', bet_amt

                if rank == 5:
                    if texture in ("wet", "semi"):
                        bet_amt = int(pot_size * random.uniform(0.5, 0.7))
                        return valid_actions[2]["action"], bet_amt
                    else:
                        bet_amt = int(pot_size * random.uniform(0.6, 0.8))
                        return valid_actions[2]["action"], bet_amt

                if rank in (3,4) or (rank == 2 and self._is_top_pair(hole_card, community)):
                    if win_mc < pot_odds + margin and self._can_fold(round_state):
                        return 'fold', 0
                    # Thin value：0.45–0.55 pot
                    bet_amt = self._clamp(int(pot_size * random.uniform(0.45,0.55)), min_r, max_r)
                    return 'raise', bet_amt

                # 5) 其他牌
                if win_mc >= pot_odds + margin:
                    return 'call', call_amt
                else:
                    return 'fold', 0
            
            else:  # light bet
                print(f"[decide_flop] call_amt={call_amt}, light bet")
                if rank >= 6:
                    bet_amt = int(pot_size * random.uniform(0.7, 0.9))
                    bet_amt = self._clamp(bet_amt, min_r, max_r)
                    return valid_actions[2]["action"], bet_amt
                elif rank == 5:
                    if texture in ("dry", "very_dry"):
                        bet_amt = int(pot_size * random.uniform(0.4, 0.6))
                        bet_amt = self._clamp(bet_amt, min_r, max_r)
                        return valid_actions[2]["action"], bet_amt
                    else:  # wet / semi
                        return valid_actions[1]["action"], call_amt  # call
                elif rank in (3, 4):
                    if texture in ("dry", "very_dry"):
                        bet_amt = int(pot_size * random.uniform(0.3, 0.5))
                        bet_amt = self._clamp(bet_amt, min_r, max_r)
                        return valid_actions[2]["action"], bet_amt
                    else:  # wet / semi
                        return valid_actions[1]["action"], call_amt  # call
                elif rank in (1, 2):
                    # rank 1, 2 都是弱牌
                    if (win_mc < 0.65 or win_mc < pot_odds + margin) and self._can_fold(round_state):
                        self.raise_fold += 1
                        return valid_actions[0]["action"], 0
                    else:
                        return valid_actions[1]["action"], call_amt  # call
                else: # rank 0
                    if self._has_strong_draw(hole_card, community):
                        return valid_actions[2]["action"], max_r  # 強抽，下注最大
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

        iterations = 2000
        win_mc, _, _ = self.estimate_winrate_mc(hole_card, community, iterations)
          

        texture = self._board_texture(community)
        print(f"[Turn] win_mc={win_mc:.2f}, rank={rank}, texture={texture}")  

        # 2) 讀池＆計算 Pot Odds
        min_r = valid_actions[2]["amount"]["min"]
        max_r = valid_actions[2]["amount"]["max"]
        pot_size = round_state["pot"]["main"]["amount"]
        call_amt = valid_actions[1]["amount"]
        pot_odds = call_amt / (pot_size + call_amt) if call_amt > 0 else 0
        stack_eff = min(s["stack"] for s in round_state["seats"]
                        if s["state"] == "participating")
        spr = stack_eff / pot_size if pot_size > 0 else 0
        margin   = 0.03  # 安全邊際

        if call_amt > 0:
            print(f"[decide_turn] passive player")
            if call_amt >= 0.6 * pot_size or call_amt > 0.40 * stack_eff:
                print(f"[decide_turn] call_amt={call_amt}, pot_size={pot_size} heavy bet")
                if rank in (8, 7):  # shove
                    print(f"[decide_turn] rank={rank}, heavy bet, shove")
                    bet_amt = max_r
                    if spr > 2:
                        pct = random.uniform(1.2, 1.6)
                        bet_amt = int(pot_size * pct)
                        bet_amt = self._clamp(bet_amt, min_r, max_r)
                    return valid_actions[2]["action"], bet_amt

                if rank == 6:
                    pct = random.uniform(1.0, 1.2)
                    bet_amt = int(pot_size * pct)
                    bet_amt = self._clamp(bet_amt, min_r, max_r)
                    return valid_actions[2]["action"], bet_amt

                elif rank == 5:
                    if texture in ("dry, very_dry"):
                        pct = random.uniform(0.6, 0.8)
                        bet_amt = int(pot_size * pct)
                        bet_amt = self._clamp(bet_amt, min_r, max_r)
                        return valid_actions[2]["action"], bet_amt
                    else:  # wet / semi
                        return valid_actions[1]["action"], call_amt  # call
                
                elif rank in (3, 4):
                    print(f"[decide_turn] rank={rank}, texture={texture} pot_odds={pot_odds:.2f}, win_mc={win_mc:.2f}")
                    if win_mc < pot_odds + margin and self._can_fold(round_state):
                        self.raise_fold += 1
                        return valid_actions[0]["action"], 0
                    elif texture in ("wet", "semi"):
                        return valid_actions[1]["action"], call_amt  # call
                    else:
                        bet_amt = int(pot_size * 0.2 + call_amt)
                        bet_amt = self._clamp(bet_amt, min_r, max_r)
                        return valid_actions[2]["action"], bet_amt

                elif rank in (1, 2):
                    if win_mc < pot_odds + margin and self._can_fold(round_state):
                        self.raise_fold += 1
                        return valid_actions[0]["action"], 0
                    elif texture in ("wet", "semi"):
                        r = random.random()
                        if r < 0.5:
                            bet_amt = int(pot_size * 2.5)
                            bet_amt = self._clamp(bet_amt, min_r, max_r)
                            return valid_actions[2]["action"], bet_amt
                        else:
                            
                            return valid_actions[1]["action"], call_amt
                    else:
                        return valid_actions[1]["action"], call_amt  # call
                
                else:
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
                    bet_amt = self._clamp(bet_amt, min_r, max_r)
                    return valid_actions[2]["action"], bet_amt

                elif rank == 5:
                    if texture in ("dry, very_dry"):
                        pct = random.uniform(0.5, 0.6)
                        bet_amt = int(pot_size * pct)
                        bet_amt = self._clamp(bet_amt, min_r, max_r)
                        return valid_actions[2]["action"], bet_amt
                    else:  # wet / semi
                        return valid_actions[1]["action"], call_amt  # call
                
                elif rank in (3, 4):
                    if texture in ("wet", "semi") and win_mc < 0.85 and self._can_fold(round_state):
                        self.raise_fold += 1
                        return valid_actions[0]["action"], 0
                    else:
                        pct = random.uniform(0.3, 0.5)
                        bet_amt = int(pot_size * pct)
                        bet_amt = self._clamp(bet_amt, min_r, max_r)
                        return valid_actions[2]["action"], bet_amt

                elif rank in (1, 2):
                    print(f"[decide_turn] texture={texture}, win_mc={win_mc:.2f}, pot_odds={pot_odds:.2f}")
                    if (win_mc < pot_odds + margin or texture == "wet") and self._can_fold(round_state):
                        self.raise_fold += 1
                        return valid_actions[0]["action"], 0
                    else:
                        return valid_actions[1]["action"], call_amt
                else:  # rank == 0
                    print(f"[decide_turn] AK? {any(c[1] in ('A', 'K') for c in hole_card)}")
                    if self._has_strong_draw(hole_card, community) and any(c[1] in ('A', 'K') for c in hole_card): # has A or K
                        print(f"[decide_turn] 強抽")
                        pct = random.uniform(0.5, 0.6)
                        bet_amt = int(pot_size * pct)
                        bet_amt = self._clamp(bet_amt, min_r, max_r)
                        return valid_actions[2]["action"], bet_amt

                    elif self._can_fold(round_state) and win_mc < 0.5:
                        self.raise_fold += 1
                        return valid_actions[0]["action"], 0
                    else:
                        return valid_actions[1]["action"], call_amt

        else:  # call_amt == 0, active player
            print("[decide_turn] call_amt == 0, active player")
            if rank >= 6: # value-bet
                print("[decide_turn] good hand, value-bet")
                pct = 0.75 if texture == "wet" else 0.85
                bet_amt = int(pot_size * pct)
                bet_amt = self._clamp(bet_amt, min_r, max_r)
                return valid_actions[2]["action"], bet_amt

            elif rank >= 4:
                print("[decide_turn] 中等牌 thin value-bet")
                pct = 0.5 if texture in ("wet", "semi") else 0.4
                bet_amt = self._clamp(int(pot_size * pct), min_r, max_r)
                return valid_actions[2]["action"], bet_amt
            
            elif rank < 2 and self._has_strong_draw(hole_card, community):
                print("[decide_turn] 強抽")
                pct = 0.6 if texture == 'wet' else 0.45
                bet_amt = self._clamp(int(pot_size * pct), min_r, max_r)
                return valid_actions[2]["action"], bet_amt

            elif rank < 2 and self._is_top_pair(hole_card, community):
                print("[decide_turn] 邊緣頂對，block bet")
                pct = 0.25
                bet_amt = self._clamp(int(pot_size * pct), min_r, max_r)
                return valid_actions[2]["action"], bet_amt

            elif rank in (1, 2):
                if win_mc > 0.75:
                    bet = self._clamp(int(pot_size * 0.60), min_r, max_r)
                    print(f"[decide_turn] rank={rank}, win_mc={win_mc:.2f}>0.75, bet={bet}")
                    return valid_actions[2]["action"], bet
                else:
                    return valid_actions[1]["action"], call_amt  # free check

            else:
                print(f"[decide_turn] free check")
                return valid_actions[1]["action"], call_amt  # free check

    def decide_river(self, valid_actions, hole_card, round_state):
        street = round_state["street"]
        community = round_state["community_card"]
        # 1) 直接計算精準勝率
        win_r, tie_r = self._calc_winrate_river(hole_card, community)
        print(f"[decide_river] win_mc={win_r:.2f}, community={community}")

        # 3) 讀池＆計算 Pot Odds
        min_r   = valid_actions[2]["amount"]["min"]
        max_r   = valid_actions[2]["amount"]["max"]
        pot_size = round_state["pot"]["main"]["amount"]
        call_amt = valid_actions[1]["amount"]
        pot_odds = call_amt / (pot_size + call_amt) if call_amt > 0 else 0
        margin   = 0.03  # 安全邊際
        print(f"[decide_river] pot_size={pot_size}, call_amt={call_amt}, pot_odds={pot_odds:.2f}")
        stack_eff = min(s["stack"] for s in round_state["seats"]
                        if s["state"] == "participating")
        spr = stack_eff / pot_size if pot_size > 0 else 0

        # —— River bluff-catch 規則 ——
        # ---------- 危險牌面 + Blocker Bonus ----------
        danger_straight = self._board_four_to_straight(community)
        danger_flush    = self._board_four_flush(community)
        texture = self._board_texture(community)

        block_straight  = self._has_straight_blocker(hole_card, community)
        block_flush     = self._has_flush_blocker(hole_card, community)
        print(f"[decide_river] danger_straight={danger_straight}, danger_flush={danger_flush}, block_straight={block_straight}, block_flush={block_flush}")

        # my card
        best   = best_of_seven(hole_card, community)
        rank   = best[0]        # 0: high-card … 8: straight-flush
        

        # adjusted win rate
        danger_penalty  = (0.10 if danger_straight else 0) + (0.08 if danger_flush else 0)
        blocker_bonus = 0.05 * (block_flush or block_straight)
        adj_win = win_r + tie_r + blocker_bonus - danger_penalty
        print(f"[decide_river] rank={rank}, adj_win={adj_win:.2f}, danger_penalty={danger_penalty:.2f}, blocker_bonus={blocker_bonus:.2f}")

        if call_amt > 0:
            print(f"[decide_river] call_amt > 0, passive player")
            if call_amt >= 0.4 * pot_size:
                print(f"[decide_river] call_amt={call_amt}, pot_size={pot_size} heavy bet")
                if rank == 8: # shove
                    print(f"[decide_river] rank={rank}, shove")
                    bet_amt = max_r
                    return valid_actions[2]["action"], bet_amt

                elif rank == 7:
                    bet_amt = max_r if spr <= 1.5 else int(pot_size * random.uniform(1.0, 1.5))
                    return valid_actions[2]["action"], bet_amt

                elif rank == 6:
                    if texture == "wet":
                        bet_amt = int((1.2 + block_flush + block_straight) * pot_size)
                    if texture == "semi":
                        bet_amt = int((1.0 + block_flush + block_straight) * pot_size)
                    else:  # dry / very_dry
                        bet_amt = int(0.8 * pot_size)

                    bet_amt = self._clamp(bet_amt, min_r, max_r)

                elif rank == 5:
                    if texture in ("wet", "semi"):
                        return valid_actions[1]["action"], call_amt  # call
                    else:
                        bet_amt = int(pot_size * random.uniform(0.6, 0.8))
                        return valid_actions[2]["action"], bet_amt

                elif rank in (3, 4):
                    if adj_win < pot_odds - margin and self._can_fold(round_state):
                        print(f"win rate too low, fold")
                        self.raise_fold += 1
                        return valid_actions[0]["action"], 0 # fold
                    if danger_flush:
                        return valid_actions[1]["action"], call_amt  # call

                    if texture in ("wet", "semi"):
                        bet_amt = int(pot_size * random.uniform(0.20, 0.25))
                    else:
                        bet_amt = int(pot_size * random.uniform(0.30, 0.35))
                    bet_amt = self._clamp(bet_amt, min_r, max_r)
                    return valid_actions[2]["action"], bet_amt
                
                elif rank in (0, 1, 2):
                    print(f"[decide_river] rank={rank}, adj_win={adj_win:.2f}, pot_odds={pot_odds:.2f}, texture={texture}")
                    if (adj_win < 0.75 or adj_win < pot_odds - margin) and self._can_fold(round_state):
                        print(f"win rate too low, fold")
                        self.raise_fold += 1
                        return valid_actions[0]["action"], 0
                    else:
                        return valid_actions[1]["action"], call_amt  # call
            
            else:  # light bet
                print(f"[decide_river] call_amt={call_amt}, pot_size={pot_size} light bet")

                # ── 1. Monster / 強價值 ───────────────────────────────
                if   rank == 8:                 # Straight-flush / Quads
                    print("[decide_river-LB] Rank 8 → pot-raise")
                    bet_amt = self._clamp(int(pot_size * 1.2), min_r, max_r)
                    return valid_actions[2]["action"], bet_amt

                elif rank == 7:                 # Full House
                    bet_amt = self._clamp(int(pot_size * 1.1), min_r, max_r)
                    return valid_actions[2]["action"], bet_amt

                elif rank == 6:                 # Nut Flush / Nut Straight
                    bet_amt = self._clamp(int(pot_size * 0.8), min_r, max_r)
                    return valid_actions[2]["action"], bet_amt

                # ── 2. 中強 (Trips / 強兩對) ──────────────────────────
                elif rank == 5:
                    # light bet → 直接加到 0.6-0.7 pot，防止對手看廉價攤牌
                    bet_amt = self._clamp(int(pot_size * random.uniform(0.60, 0.70)),
                                        min_r, max_r)
                    return valid_actions[2]["action"], bet_amt

                elif rank in (3, 4):            # Two-pair / Over-pair
                    # 計算 bluff-catch EV
                    if adj_win < pot_odds - margin and self._can_fold(round_state):
                        print("[decide_river-LB] Over-pair / Two-pair but EV<0 → fold")
                        self.raise_fold += 1
                        return valid_actions[0]["action"], 0
                    else:
                        # 半 pot  thin-value 或 block-bet 回擊
                        if texture in ("wet", "semi"):
                            bet_amt = self._clamp(int(pot_size * random.uniform(0.35, 0.45)),
                                                min_r, max_r)
                            return valid_actions[2]["action"], bet_amt
                        else:
                            # 乾板直接 call，避免被輕鬆 3-bet
                            return valid_actions[1]["action"], call_amt

                # ── 3. 弱牌 / 單對以下 ──────────────────────────────
                elif rank in (1, 2):                           # rank 0-2
                    print(f"[decide_river-LB] rank={rank}, adj={adj_win:.2f}, pot_odds={pot_odds:.2f}")
                    if adj_win >= pot_odds + margin and adj_win >= 0.7:
                        # +EV 接注 → bluff-catch
                        return valid_actions[1]["action"], call_amt
                    if self._can_fold(round_state):
                        self.raise_fold += 1
                        return valid_actions[0]["action"], 0
                    else:
                        return valid_actions[1]["action"], call_amt
                
                else:  # rank == 0
                    print(f"[decide_river-LB] rank={rank}, adj={adj_win:.2f}, pot={pot_odds:.2f}")
                    if self._can_fold(round_state):
                        self.raise_fold += 1
                        return valid_actions[0]["action"], 0
                    else:
                        return valid_actions[1]["action"], call_amt


        else:
            print(f"[decide_river] active: rank={rank}, adj_win={adj_win:.2f}, pot_odds={pot_odds:.2f}, texture={texture}")          

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
            def bet_by_pct(pct_lo, pct_hi):
                pct = random.uniform(pct_lo, pct_hi)
                bet = int(pot_size * pct)
                return max(min_r, min(bet, max_r))
            # === A. 超強牌（rank 8）→ 大注 / Over-bet ===
            if rank == 8:
                bet_amt = bet_by_pct(0.9, 2.2 if texture in ("semi", "wet") else 1.4)
                return valid_actions[2]["action"], bet_amt

            # === B. Very Strong（rank 7）====
            elif rank == 7:
                bet_amt = bet_by_pct(0.75, 1.2 if texture in ("semi", "wet") else 1.0)
                return valid_actions[2]["action"], bet_amt

            # === C. Strong Value（rank 6）====
            elif rank == 6:
                bet_amt = bet_by_pct(0.55, 0.85)
                return valid_actions[2]["action"], bet_amt

            # === D. 中強（Trips，rank 5）====
            elif rank == 5:
                bet_amt = bet_by_pct(0.40, 0.60)
                return valid_actions[2]["action"], bet_amt

            # === E. 中檔（Two-Pair / Over-Pair，rank 3-4）====
            elif rank in (3, 4):
                if danger_penalty > 0:                     # 濕板 → 偏小 block
                    bet_amt = bet_by_pct(0.25, 0.35)
                else:
                    bet_amt = bet_by_pct(0.35, 0.55)
                return valid_actions[2]["action"], bet_amt

            # === F. 頂對 / 中對（rank 2）或更差 ===
            #     - 若對手下注，進入 bluff-catch 判斷
            #     - 否則可嘗試 20-30% pot 輕偷或 simply check
            print(f"[decide_river] rank={rank}, F.")
            if random.random() < 0.7 and texture in ("dry", "very_dry"):
                bet_amt = bet_by_pct(0.20, 0.30)
                return valid_actions[2]["action"], bet_amt
            elif random.random() < 0.7 and texture in ("semi"):
                bet_amt = bet_by_pct(0.20, 0.25)
                return valid_actions[2]["action"], bet_amt
            return valid_actions[1]["action"], 0            # check


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
  
        return act, amt

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
