import random
import time
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
        self.mc_iters = 8000  # 每次模擬的迭代次數
        self.mc_time_limit = 4.0  # 每次模擬的時間限制（秒）

        self.preflop_thresholds = {
            "early":  {"raise_big": 0.75, "raise_small": 0.70, "call": 0.48},
            "middle": {"raise_big": 0.70, "raise_small": 0.65, "call": 0.45},
            "late":   {"raise_big": 0.65, "raise_small": 0.50, "call": 0.30},
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
        # 0) 遭遇 limp ⇒ Isolation Raise
        # =========================================================
        # 條件：對手只有 limp (call_amt == bb) ；自己還能 raise；手牌達到某 winrate
        if can_raise and call_amt == bb:
            # 以 Seat 位置設定 Iso-Raise 閾值（比普通 raise_small 再寬一點）
            iso_thresh = thr["raise_small"] - 0.05      # 例：late 0.50 → 0.45
            if winrate >= iso_thresh:
                # Isolation Size：2.5–3.5 BB 隨機
                factor   = random.uniform(2.5, 3.5)
                desired  = int(bb * factor)
                bet_amt  = max(min_r, min(desired, max_r))
                return valid_actions[2]["action"], bet_amt
            # 若手牌太差 → 直接 Check（若在 BB）或 Fold（BTN）
            if seat_idx == sb_pos:     # SB limp back
                print("[Preflop] SB limp, check allowed")
                return valid_actions[1]["action"], call_amt
            else:                      # BB 對 BTN limp → 允許 check
                print("[Preflop] BTN limp, fold")
                return valid_actions[0]["action"], 0

        # =========================================================
        # 1) 強牌：3-bet／shove
        # =========================================================
        if can_raise and winrate > thr["raise_big"]:
            bet_amt = max(min_r, max_r)                  # 最大壓力
            print(f"[Preflop] 強牌 3-bet/shove {bet_amt}")
            return valid_actions[2]["action"], bet_amt

        # =========================================================
        # 2) 依對手加注大小微調 call 閾值
        # =========================================================
        stack_eff = min(p["stack"] for p in seats if p["state"]=="participating")
        pot_odds  = call_amt / (pot + call_amt) if call_amt else 0
        margin    = 0.02

        if call_amt >= 8*bb or call_amt > 0.25*stack_eff:
            call_thresh = max(thr["raise_small"], pot_odds + margin)
        elif call_amt >= 4*bb:
            call_thresh = max(thr["call"] + 0.05, pot_odds + margin)
        else:
            call_thresh = max(thr["call"], pot_odds + margin)

        # =========================================================
        # 3) 決定 Call / Fold
        # =========================================================
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
    rank_order = "23456789TJQKA"

    def _is_top_pair(self, hole, board):
        """判斷是否擊中頂對 (手牌任一張 = board 最高 rank)"""
        if not board: 
            return False
        board_high = max(board, key=lambda c: rank_order.index(c[1]))[1]
        return any(c[1] == board_high for c in hole)

    def _has_strong_draw(self, hole, board):
        """OESD or 9 outs Flush draw"""
        flush_draw, straight_draw, outs = self._detect_draws(hole, board)
        return (flush_draw or straight_draw) and outs >= 8   # 8–9 outs

    def _board_four_to_straight(board):
        """Flop/Turn/River 是否有四張連號"""
        vals = sorted(set(rank_order.index(c[1]) for c in board))
        # 考慮 Wheel
        if 12 in vals: vals.append(-1)
        return any(vals[i+3]-vals[i]==3 for i in range(len(vals)-3))

    def _board_three_flush(board):
        suits = [c[0] for c in board]
        return any(suits.count(s) >= 3 for s in set(suits))

    def decide_flop(self, valid_actions, hole_card, round_state):
        community = round_state["community_card"]
        win_mc, _, _ = self.estimate_winrate_mc(hole_card, community, iterations=5000)
        texture = self._board_texture(community)

        # 1) 計算 pot size & call amount
        pot_size   = round_state["pot"]["main"]["amount"]
        min_r = valid_actions[2]["amount"]["min"]
        max_r = valid_actions[2]["amount"]["max"]
        call_amt   = valid_actions[1]["amount"]
        print(f"[Flop] pot_size={pot_size}, call_amt={call_amt}, texture={texture}")
        pot_odds   = call_amt / (pot_size + call_amt)
        margin = 0.03
        implied = self._calc_implied(round_state, pot_size)
        print(f"[Flop] win_mc={win_mc:.2f}, pot_odds={pot_odds:.2f}")    

        # 強抽+頂對強制下注
        if self._is_top_pair(hole_card, community) and self._has_strong_draw(hole_card, community):
            pct = 0.55 if texture in ("wet","semi") else 0.45   # 濕板收 55 %, 乾板 45 %
            bet = self._clamp(int(pot_size * pct), min_r, max_r)
            return valid_actions[2]["action"], bet
 

        # 2) 決策：若勝率 <= pot_odds → fold
        # 若勝率 > pot_odds + margin → 下注或加注
        if win_mc <= pot_odds + margin:
            # 先檢查是否有抽牌且值得半閃
            flush_draw, straight_draw, outs = self._detect_draws(hole_card, community)
            draw_equity = outs / (52 - len(hole) - len(community))
            if (flush_draw or straight_draw) and (draw_equity + implied > pot_odds + margin):
                # wet or dry
                if texture == "wet":
                    bet_amt = max(min_r, min(max_r, int(pot_size * 0.3)))
                else:
                    bet_amt = max(min_r, min(max_r, int(pot_size * 0.22)))
                print(f"[Flop] 半閃下注 {bet_amt} (30% pot)")
                return valid_actions[2]["action"], bet_amt
            # 否則棄牌
            print("[Flop] 棄牌")
            return valid_actions[0]["action"], 0

        if win_mc > 0.70:
            # 強牌：下注 90% pot（或全下）
            if texture == "wet":
                r = random.random() * 0.20 + 0.7   # 0.70 ~ 0.90
            elif texture == "semi":
                r = random.random() * 0.10 + 0.6   # 0.60 ~ 0.70
            else:           # dry / very_dry
                r = random.random() * 0.10 + 0.50  # 0.50 ~ 0.60
            bet = max(min_r, min(int(pot_size * r), max_r))
            print(f"[Flop] 強牌下注 {bet} (90% pot)")
            return valid_actions[2]["action"], bet

        else:  # 中弱牌
            # 乾板可用小偷池；濕板就跟注或過牌
            if texture.startswith("dry"):
                steal = max(min_r, min(max_r, int(pot_size * 0.25)))
                print(f"[Flop] 小偷池 {steal} (25% pot)")
                return valid_actions[2]["action"], steal
            else:
                # 沒人下注可 check，否則 call
                if call_amt == 0:
                    print("[Flop] free check")
                    return valid_actions[1]["action"], 0
                else:
                    print(f"[Flop] 跟注 {call_amt}")
                    return valid_actions[1]["action"], call_amt
    
    def decide_turn(self, valid_actions, hole_card, round_state):
        street = round_state["street"]
        community = round_state["community_card"]
        # 1) 進階 Equity Simulation
        #    Turn 用預設 mc_iters，River 用少量 MC 快速估

        iterations = 2000 if street == "turn" else 1000
        win_mc, _, _ = self.estimate_winrate_mc(hole_card, community, iterations)


        # 2) 抽牌檢測（供後面半閃 & check‐raise 用）    
        flush_draw, straight_draw, outs = self._detect_draws(hole_card, community)

        # 3) 讀池＆計算 Pot Odds
        pot_size = round_state["pot"]["main"]["amount"]
        call_amt = valid_actions[1]["amount"]
        pot_odds = call_amt / (pot_size + call_amt) if call_amt > 0 else 0
        margin   = 0.03  # 安全邊際

        if call_amt == 0 and self._is_top_pair(hole_card, community) and self._has_strong_draw(hole_card, community):
            print("[decide_turn] 強抽+頂對強制下注")
            texture = self._board_texture(community)
            pct = 0.55 if texture in ("wet", "semi") else 0.45   # 濕板55%、乾板45%
            bet_amt = int(pot_size * pct)
            bet_amt = max(min_r, min(bet_amt, max_r))
            return valid_actions[2]["action"], bet_amt

        # 4) 看是否面臨對手大注，可考慮 Check‐Raise
        #    當對手下注 (call_amt>0)，且我們有抽牌或中檔成手，就先反擊
        best = best_of_seven(hole_card, community)
        rank = best[0]
        min_r   = valid_actions[2]["amount"]["min"]
        max_r   = valid_actions[2]["amount"]["max"]
        if call_amt > 0.4 * pot_size and (flush_draw or straight_draw or (3 <= rank < 5)):
            # 用最小加注量或 20% pot 當 Check‐Raise 大小
            raise_amt = int(pot_size * 0.2 + call_amt)
            raise_amt = max(min_r, min(raise_amt, max_r))
            return valid_actions[2]["action"], raise_amt

        # 5) 長期虧錢時，嘗試 Block‐Bet（中檔以下）或棄
        if win_mc <= pot_odds + margin:
            # 有抽牌時做半閃
            draw_equity = outs / (52 - len(hole_card) - len(community))
            implied     = self._calc_implied(round_state, pot_size)
            if (flush_draw or straight_draw) and (draw_equity + implied > pot_odds + margin):
                # 半閃：下注 30% pot
                bet_amt = int(pot_size * 0.3)
                bet_amt = max(min_r, min(bet_amt, max_r))
                return valid_actions[2]["action"], bet_amt
            # 否則棄牌
            return valid_actions[0]["action"], 0

        # 6) Value‐Bet & Thin Value‐Bet & Block‐Bet
        def _rand_pct(lo, hi):
            return random.uniform(lo, hi)

        texture = self._board_texture(community)
        print(f"[decide_turn_river] rank={rank}, win_mc={win_mc:.2f}, pot_odds={pot_odds:.2f}, texture={self._board_texture(community)}")
        if rank >= 7:
            pct = {
                "wet"      : _rand_pct(0.85, 1.00),
                "semi"     : _rand_pct(0.75, 0.95),
                "dry"      : _rand_pct(0.65, 0.80),
                "very_dry" : _rand_pct(0.60, 0.75)
            }[texture]

            bet_amt = int(pot_size * pct)
            bet_amt = max(min_r, min(bet_amt, max_r))
            return valid_actions[2]["action"], bet_amt

        # -------- 中強牌 (rank 4~6) ----------
        elif rank >= 4:
            pct = {
                "wet"      : _rand_pct(0.65, 0.80),
                "semi"     : _rand_pct(0.60, 0.75),
                "dry"      : _rand_pct(0.50, 0.60),
                "very_dry" : _rand_pct(0.45, 0.55)
            }[texture]

            bet_amt = int(pot_size * pct)
            bet_amt = max(min_r, min(bet_amt, max_r))
            return valid_actions[2]["action"], bet_amt

        elif rank >= 2:
            # 中檔（Three‐of‐a‐Kind / Two‐Pair）
            if texture == "wet":
                pct = random.uniform(0.55, 0.65)
            elif texture == "semi":
                pct = random.uniform(0.45, 0.55)
            else:  # dry / very_dry
                pct = random.uniform(0.35, 0.45)

            bet_amt = int(pot_size * pct)
            bet_amt = max(min_r, min(bet_amt, max_r))  # clamp 合法範圍
            if min_r > max_r:                          # 已無法再 raise
                return valid_actions[1]["action"], valid_actions[1]["amount"]  # call
            return valid_actions[2]["action"], bet_amt

        # 7) 剩下就 Check 或 Fold
        print(f"[decide_turn] call_amt={call_amt}")
        if call_amt == 0:
            # free check
            return valid_actions[1]["action"], 0
        if win_mc >= pot_odds + margin:
            return valid_actions[1]["action"], call_amt  # call

        return valid_actions[0]["action"], 0

    def decide_river(self, valid_actions, hole_card, round_state):
        street = round_state["street"]
        community = round_state["community_card"]
        # 1) 進階 Equity Simulation
        #    Turn 用預設 mc_iters，River 用少量 MC 快速估

        iterations = 2000 if street == "turn" else 1000
        win_mc, _, _ = self.estimate_winrate_mc(hole_card, community, iterations)


        # 2) 抽牌檢測（供後面半閃 & check‐raise 用）    
        flush_draw, straight_draw, outs = self._detect_draws(hole_card, community)

        # 3) 讀池＆計算 Pot Odds
        pot_size = round_state["pot"]["main"]["amount"]
        call_amt = valid_actions[1]["amount"]
        pot_odds = call_amt / (pot_size + call_amt) if call_amt > 0 else 0
        margin   = 0.03  # 安全邊際

        # —— River bluff-catch 規則 ——
        if street == "river":
            danger_straight = self._board_four_to_straight(community)
            danger_flush    = self._board_three_flush(community)

            if (danger_straight or danger_flush) and rank <= 2:   # 單對 / 兩對
                print(f"[decide_river] 危險牌面，rank={rank}, win_mc={win_mc:.2f}, pot_odds={pot_odds:.2f}")
                if call_amt == 0:
                    return valid_actions[1]["action"], 0          # check
                # 只有在 +EV 時跟注；否則棄
                if win_mc >= pot_odds + 0.03:
                    return valid_actions[1]["action"], call_amt   # bluff-catch
                else:
                    return valid_actions[0]["action"], 0          # fold


        # 4) 看是否面臨對手大注，可考慮 Check‐Raise
        #    當對手下注 (call_amt>0)，且我們有抽牌或中檔成手，就先反擊
        best = best_of_seven(hole_card, community)
        rank = best[0]
        min_r   = valid_actions[2]["amount"]["min"]
        max_r   = valid_actions[2]["amount"]["max"]
        if call_amt > 0.4 * pot_size and (flush_draw or straight_draw or (3 <= rank < 5)):
            # 用最小加注量或 20% pot 當 Check‐Raise 大小
            raise_amt = int(pot_size * 0.2 + call_amt)
            raise_amt = max(min_r, min(raise_amt, max_r))
            return valid_actions[2]["action"], raise_amt

        # 5) 長期虧錢時，嘗試 Block‐Bet（中檔以下）或棄
        if win_mc <= pot_odds + margin:
            # 有抽牌時做半閃
            draw_equity = outs / (52 - len(hole_card) - len(community))
            implied     = self._calc_implied(round_state, pot_size)
            if (flush_draw or straight_draw) and (draw_equity + implied > pot_odds + margin):
                # 半閃：下注 30% pot
                bet_amt = int(pot_size * 0.3)
                bet_amt = max(min_r, min(bet_amt, max_r))
                return valid_actions[2]["action"], bet_amt
            # 否則棄牌
            return valid_actions[0]["action"], 0

        # 6) Value‐Bet & Thin Value‐Bet & Block‐Bet
        def _rand_pct(lo, hi):
            return random.uniform(lo, hi)

        texture = self._board_texture(community)
        print(f"[decide_turn_river] rank={rank}, win_mc={win_mc:.2f}, pot_odds={pot_odds:.2f}, texture={self._board_texture(community)}")
        if rank >= 7:
            pct = {
                "wet"      : _rand_pct(0.90, 2.00),   # 可 over-bet
                "semi"     : _rand_pct(0.80, 1.00),
                "dry"      : _rand_pct(0.70, 0.90),
                "very_dry" : _rand_pct(0.65, 0.85)
            }[texture]

            bet_amt = int(pot_size * pct)
            bet_amt = max(min_r, min(bet_amt, max_r))
            return valid_actions[2]["action"], bet_amt

        # -------- 中強牌 (rank 4~6) ----------
        elif rank >= 4:
            pct = {
                "wet"      : _rand_pct(0.60, 0.80),
                "semi"     : _rand_pct(0.55, 0.70),
                "dry"      : _rand_pct(0.45, 0.60),
                "very_dry" : _rand_pct(0.40, 0.55)
            }[texture]

            bet_amt = int(pot_size * pct)
            bet_amt = max(min_r, min(bet_amt, max_r))
            return valid_actions[2]["action"], bet_amt

        elif rank >= 2:
            # 中檔（Three‐of‐a‐Kind / Two‐Pair）
            if texture == "wet":
                pct = random.uniform(0.25, 0.35)   # Block-Bet
            elif texture == "semi":
                pct = random.uniform(0.30, 0.40)
            else:                                  # dry / very_dry
                pct = random.uniform(0.40, 0.50)

            bet_amt = int(pot_size * pct)
            bet_amt = max(min_r, min(bet_amt, max_r))  # clamp 合法範圍
            if min_r > max_r:                          # 已無法再 raise
                return valid_actions[1]["action"], valid_actions[1]["amount"]  # call
            return valid_actions[2]["action"], bet_amt

        # 7) 剩下就 Check 或 Fold
        print(f"[decide_turn_river] call_amt={call_amt}")
        if call_amt == 0:
            # free check
            return valid_actions[1]["action"], 0
        if win_mc >= pot_odds + margin:
            return valid_actions[1]["action"], call_amt  # call

        return valid_actions[0]["action"], 0

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
