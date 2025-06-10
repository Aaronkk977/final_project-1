import random
import time
import itertools
import traceback
import torch
from game.players import BasePokerPlayer
from solver import best_of_seven, hand_rank
from deck import Deck

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
        self.preflop_table = self.load_preflop_csv("agents/preflop_equity_2000.csv")
        self._rank_order = "23456789TJQKA"
        self._suit = "CDHS"
        self._deck = Deck().full_deck()  
        self.mc_iters = 8000  # 每次模擬的迭代次數
        self.mc_time_limit = 4.0  # 每次模擬的時間限制（秒）

        self.preflop_thresholds = {
            "early":  {"raise_big": 0.70, "raise_small": 0.63, "call": 0.5},
            "middle": {"raise_big": 0.70, "raise_small": 0.65, "call": 0.45},
            "late":   {"raise_big": 0.60, "raise_small": 0.55, "call": 0.30},
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

    def decide_preflop(self, valid_actions, hole, round_state):
        seats   = round_state["seats"]
        num_p   = len(seats)
        button  = round_state["dealer_btn"]
        sb_pos  = round_state["small_blind_pos"]
        seat_idx= next(i for i,s in enumerate(seats) if s["name"]=="myBot")
        pos     = self.get_position(seat_idx, button, num_p, sb_pos)
        thr     = self.preflop_thresholds[pos]

        winrate = self.preflop_table.get(normalize_hand(hole), 0)

        # ───────────── 共用變量 ─────────────
        pot      = round_state["pot"]["main"]["amount"]
        call_amt = valid_actions[1]["amount"]
        bb       = round_state["small_blind_amount"] * 2
        min_r, max_r = valid_actions[2]["amount"].values()
        can_raise    = min_r <= max_r
        #───────────────────────────────────

        # =========================================================
        # 0) 強牌：3-bet／shove
        # =========================================================
        if can_raise and winrate > thr["raise_big"]:
            # ==== 先判斷有效籌碼深度（stack_eff） ====
            stack_eff = min(p["stack"] for p in seats if p["state"] == "participating")
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

            print(f"[Preflop] Monster hand → {choice} 3-bet, bet_amt={bet_amt}")
            return valid_actions[2]["action"], bet_amt
        
        # =========================================================
        # 1) 遭遇 limp ⇒ Isolation Raise
        # =========================================================
        # 條件：對手只有 limp (call_amt == bb) ；自己還能 raise；手牌達到某 winrate
        if can_raise and call_amt == bb:
            # 以 Seat 位置設定 Iso-Raise 閾值（比普通 raise_small 再寬一點）
            print(f"[Preflop] limp call_amt={call_amt}, winrate={winrate:.2f}, position={pos}")

            iso_thresh = thr["raise_small"] - 0.03      # 例：late 0.50 → 0.45
            if winrate >= iso_thresh:
                # Isolation Size：2.5–3.5 BB 隨機
                factor   = random.uniform(2.5, 3.5)
                desired  = int(bb * factor)
                bet_amt  = max(min_r, min(desired, max_r))
                return valid_actions[2]["action"], bet_amt
    
            else:    
                print("[Preflop] card is not good, free check")
                return valid_actions[1]["action"], call_amt  # check

        # =========================================================
        # 2) 依對手加注大小微調 call 閾值，並根據閾值決定 Call / Fold
        # =========================================================
        stack_eff = min(p["stack"] for p in seats if p["state"]=="participating")
        pot_odds  = call_amt / (pot + call_amt) if call_amt else 0
        margin    = 0.02

        if call_amt >= 8 * bb or call_amt > 0.25 * stack_eff:
            call_thresh = max(thr["raise_small"], pot_odds + margin)
        elif call_amt >= 4*bb:
            call_thresh = max(thr["call"] + 0.05, pot_odds + margin)
        else:
            call_thresh = max(thr["call"], pot_odds + margin)

        if winrate >= call_thresh:
            print(f"[Preflop] Call (winrate={winrate:.2f}, call_thresh={call_thresh:.2f})")
            return valid_actions[1]["action"], call_amt
        else:
            print(f"[Preflop] Fold (winrate={winrate:.2f}, call_thresh={call_thresh:.2f})")
            return valid_actions[0]["action"], 0


    def estimate_winrate_mc(self, hole, community, iterations):
        """
        Monte Carlo 模擬勝率，支援快取與超時回退
        回傳 (win_rate, tie_rate, loss_rate)
        """
        deck_template = Deck().full_deck()
        wins = ties = 0
        start = time.time()

        for i in range(iterations):
            # 超時檢查：如果超過 time_limit，就提早跳出
            if time.time() - start > self.mc_time_limit:
                break
            # 準備牌堆，移除已知牌
            deck = [c for c in deck_template if c not in hole and c not in community]
            # 隨機抽對手洞牌
            opp = random.sample(deck, 2)
            for c in opp: deck.remove(c)
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

    def _detect_draws(self, hole, community):
        """
        回傳 (flush_draw, straight_draw, outs)
        - flush_draw: 還差 1 張同花即可成牌
        - straight_draw: open-ended 或 gutshot 都算 True
        - outs: 完成 draw 所剩牌數 (max 9)
        """
        rank_order = "23456789TJQKA"
        cards      = hole + community
        board_len  = len(community)
        remaining  = 52 - len(cards)          # 還沒曝光的牌張數

        # ---------- Flush Draw ----------
        suits = [c[0] for c in cards]
        flush_draw = False
        outs = 0
        if board_len <= 4:                    # River 再也抽不到牌，就不算 draw
            for s in set(suits):
                cnt = suits.count(s)
                if cnt == 4:                  # 還差一張＝典型同花抽
                    flush_draw = True
                    outs = max(outs, 9)
                    break                     # 取最大 outs 即可

        # ---------- Straight Draw ----------
        # 轉數字並考慮 Wheel (A-2-3-4-5)
        vals = sorted({rank_order.index(c[1]) for c in cards})
        # 把 A 當 1 再加一次，方便偵測 A-2-3-4
        if 12 in vals:                        # 12 == 'A'
            vals.append(-1)

        open_ended = gutshot = False
        for i in range(len(vals) - 3):
            window = vals[i:i+4]
            span   = window[-1] - window[0]
            if span == 3 and len(set(window)) == 4:
                open_ended = True            # 4 張連續
            elif span == 4 and len(set(window)) == 4:
                gutshot = True               # 差一張中洞
        straight_draw = open_ended or gutshot
        if open_ended:
            outs = max(outs, 8)
        elif gutshot:
            outs = max(outs, 4)

        # ---------- outs /  remaining ----------
        # 呼叫端若要用 equity，可用 outs / remaining
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

    def _is_top_pair(self, hole, board):
        """判斷是否擊中頂對 (手牌任一張 = board 最高 rank)"""
        if not board: 
            return False
        board_high = max(board, key=lambda c: self._rank_order.index(c[1]))[1]
        print(f"[is_top_pair] board_high={board_high}, hole={hole}")
        return any(c[1] == board_high for c in hole)

    def _has_strong_draw(self, hole, board):
        """OESD or 9 outs Flush draw"""
        flush_draw, straight_draw, outs = self._detect_draws(hole, board)
        print(f"[has_strong_draw] flush_draw={flush_draw}, straight_draw={straight_draw}, outs={outs}")
        return (flush_draw or straight_draw) and outs >= 8   # 8–9 outs

    def _board_four_to_straight(self, board):
        """Flop/Turn/River 是否有四張連號"""
        vals = sorted(set(self._rank_order.index(c[1]) for c in board))
        # 考慮 Wheel
        if 12 in vals: vals.append(-1)
        return any(vals[i+3]-vals[i]==3 for i in range(len(vals)-3))

    def _board_three_flush(self, board):
        suits = [c[0] for c in board]
        return any(suits.count(s) >= 3 for s in set(suits))

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
                need = {ranks[(window[0] + j) % 13] for j in range(5)}
                my_ranks = {c[1] for c in hole}
                if my_ranks & need:                 # 有交集 = blocker
                    return True
        return False

    def decide_flop(self, valid_actions, hole_card, round_state):
        community = round_state["community_card"]
        win_mc, _, _ = self.estimate_winrate_mc(hole_card, community, iterations=5000)
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
            if self._is_top_pair(hole_card, community) and self._has_strong_draw(hole_card, community):
                print("[decide_flop] 強抽+頂對強制下注")
                pct = 0.55 if texture in ("wet", "semi") else 0.45
                bet_amt = int(pot_size * pct)
                bet_amt = self._climp(bet_amt, min_r, max_r)
                return valid_actions[2]["action"], bet_amt
            elif rank >= 5:
                pct = 0.7 if texture == "wet" else 0.55
                bet_amt = int(pot_size * pct)
                bet_amt = self._clamp(bet_amt, min_r, max_r)
                return valid_actions[2]["action"], bet_amt
            else:
                print(f"[decide_flop] free check")
                return valid_actions[1]["action"], call_amt  # free check
        else: # passive
            if call_amt >= 0.4 * pot_size or call_amt > 0.25 * stack_eff: # heavy bet
                print(f"[decide_flop] call_amt={call_amt}, heavy bet")
                if rank == 8: # shove
                    bet_amt = max_r
                    return valid_actions[2]["action"], bet_amt

                elif rank == 7:
                    bet_amt = max_r if spr <= 1.5 else pot_size + call_amt
                    return valid_actions[2]["action"], bet_amt

                elif rank == 6:
                    if texture in ("wet", "semi"):
                        bet_amt = int(0.8 * pot_size)
                    else:  # dry / very_dry
                        bet_amt = int(1.2 * pot_size)

                    bet_amt = self._clamp(bet_amt, min_r, max_r)

                elif rank == 5:
                    if texture in ("wet", "semi"):
                        return valid_actions[1]["action"], call_amt  # call
                    else:
                        bet_amt = int(pot_size * random.uniform(0.6, 0.8))
                        return valid_actions[2]["action"], bet_amt

                elif rank in (3, 4):
                    if win_mc < pot_odds + margin:
                        return valid_actions[0]["action"], 0 # fold
                    else:
                        if texture in ("wet", "semi"):
                            bet_amt = int(pot_size * random.uniform(0.10, 0.15))
                        else:
                            bet_amt = int(pot_size * random.uniform(0.20, 0.25))
                        bet_amt = self._clamp(bet_amt, min_r, max_r)
                        return valid_actions[2]["action"], bet_amt
                
                elif rank in (0, 2):
                    if win_mc < pot_odds + margin:
                        return valid_actions[0]["action"], 0
                    else:
                        return valid_actions[1]["action"], call_amt  # call
            
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
                else:
                    # rank 0, 1, 2 都是弱牌
                    if win_mc < pot_odds + margin:
                        return valid_actions[0]["action"], 0
                    else:
                        return valid_actions[1]["action"], call_amt  # call

    
    def decide_turn(self, valid_actions, hole_card, round_state):
        street = round_state["street"]
        community = round_state["community_card"]
        
        best = best_of_seven(hole_card, community)
        rank = best[0]

        iterations = 2000
        win_mc, _, _ = self.estimate_winrate_mc(hole_card, community, iterations)
          
        flush_draw, straight_draw, outs = self._detect_draws(hole_card, community)
        texture = self._board_texture(community)
        print(f"[Turn] win_mc={win_mc:.2f}, rank={rank}, texture={texture}")  
        print(f"[Turn] flush_draw={flush_draw}, straight_draw={straight_draw}, outs={outs}")

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
            if call_amt >= 0.6 * pot_size or call_amt > 0.40 * stack_eff:
                if rank in (8, 7):  # shove
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
                    if win_mc < pot_odds + margin:
                        return valid_actions[0]["action"], 0
                    else:
                        bet_amt = int(pot_size * 0.25 + call_amt)
                        bet_amt = self._clamp(bet_amt, min_r, max_r)
                        return valid_actions[2]["action"], bet_amt

                elif rank in (0, 1, 2):
                    if win_mc < pot_odds + margin:
                        return valid_actions[0]["action"], 0
                    else:
                        return valid_actions[1]["action"], call_amt

            else:
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
                    if texture in ("wet", "semi"):
                        return valid_actions[0]["action"], 0
                    else:
                        pct = random.uniform(0.3, 0.5)
                        bet_amt = int(pot_size * pct)
                        bet_amt = self._clamp(bet_amt, min_r, max_r)
                        return valid_actions[2]["action"], bet_amt

                elif rank in (0, 1, 2):
                    if win_mc < pot_odds + margin:
                        return valid_actions[0]["action"], 0
                    else:
                        return valid_actions[1]["action"], call_amt


        else:  # call_amt == 0, active player
            if rank >= 6: # value-bet
                pct = 0.65 if texture == "wet" else 0.85
                bet_amt = int(pot_size * pct)
                bet_amt = self._clamp(bet_amt, min_r, max_r)
                return valid_actions[2]["action"], bet_amt
            elif self._has_strong_draw(hole_card, community):
                print("[decide_turn] 強抽，半閃")
                pct = 0.70 if texture_turn == 'wet' else 0.60
                bet = clamp(int(pot * pct), min_r, max_r)
                return valid_actions[2]["action"], bet_amt
            else:
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
        danger_flush    = self._board_three_flush(community)
        texture = self._board_texture(community)

        block_straight  = self._has_straight_blocker(hole_card, community)
        block_flush     = self._has_flush_blocker(hole_card, community)

        # my card
        best   = best_of_seven(hole_card, community)
        rank   = best[0]        # 0: high-card … 8: straight-flush
        

        # adjusted win rate
        danger_penalty  = (0.10 if danger_straight else 0) + (0.07 if danger_flush else 0)
        blocker_bonus = 0.05 * (block_flush or block_straight)
        adj_win = win_r + tie_r + blocker_bonus - danger_penalty

        if call_amt > 0:
            print(f"[decide_river] call_amt > 0, passive player")
            if call_amt >= 0.4 * pot_size:

                if rank == 8: # shove
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
                    if adj_win < pot_odds - margin:
                        print(f"win rate too low, fold")
                        return valid_actions[0]["action"], 0 # fold
                    else:
                        if texture in ("wet", "semi"):
                            bet_amt = int(pot_size * random.uniform(0.20, 0.25))
                        else:
                            bet_amt = int(pot_size * random.uniform(0.30, 0.35))
                        bet_amt = self._clamp(bet_amt, min_r, max_r)
                        return valid_actions[2]["action"], bet_amt
                
                elif rank in (0, 1, 2):
                    if adj_win < pot_odds + margin:
                        print(f"win rate too low, fold")
                        return valid_actions[0]["action"], 0
                    else:
                        return valid_actions[1]["action"], call_amt  # call


        else:
            print(f"[decide_river] normal bet or non. rank={rank}, adj_win={adj_win:.2f}, pot_odds={pot_odds:.2f}, texture={texture}")          

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
                    bet_amt = bet_by_pct(0.35, 0.50)
                return valid_actions[2]["action"], bet_amt

            # === F. 頂對 / 中對（rank 2）或更差 ===
            #     - 若對手下注，進入 bluff-catch 判斷
            #     - 否則可嘗試 20-30% pot 輕偷或 simply check
            print(f"[decide_river] rank={rank}, F.")
            if call_amt > 0:
                # +EV 才跟；用 adj_win
                print(f"[decide_river] +EV 才跟；用 adj_win")
                if adj_win >= pot_odds + margin:
                    return valid_actions[1]["action"], call_amt
                else:
                    return valid_actions[0]["action"], 0        # fold
            else:
                # 無人下注可偶爾偷
                if random.random() < 0.25 and texture in ("dry", "very_dry"):
                    bet_amt = bet_by_pct(0.20, 0.30)
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
