import random
import itertools
import csv

# 1. 定義牌面與花色
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
suits = ['h', 'd', 'c', 's']

def full_deck():
    """回傳完整 52 張牌的清單"""
    return [r + s for r in ranks for s in suits]

# 2. 牌力評分函式（同之前示例）
rank_map = {r: i+2 for i, r in enumerate(ranks)}
def hand_rank(cards):
    vals = sorted([rank_map[c[0]] for c in cards], reverse=True)
    suits_ = [c[1] for c in cards]
    flush = len(set(suits_)) == 1 # flush 同花
    # straight and top
    unique_vals = sorted(set(vals), reverse=True)
    if unique_vals == [14, 5, 4, 3, 2]: # A-2-3-4-5
        straight, top = True, 5
    else:
        straight = len(unique_vals) == 5 and all(unique_vals[i] - 1 == unique_vals[i+1] for i in range(4))
        top = unique_vals[0] if straight else None
        
    counts = {v: vals.count(v) for v in set(vals)}
    freq = sorted(counts.values(), reverse=True)

    if straight and flush:
        return (8, top)
    if freq == [4, 1]: # 四條
        four = [v for v, c in counts.items() if c == 4][0]
        kicker = max(v for v in vals if v != four)
        return (7, four, kicker)
    if freq == [3, 2]: # 葫蘆
        three = [v for v, c in counts.items() if c == 3][0]
        pair  = [v for v, c in counts.items() if c == 2][0]
        return (6, three, pair)
    if flush:          # 同花
        return (5, *vals)
    if straight:       # 順子
        return (4, top)
    if freq == [3, 1, 1]: # 三條
        three = [v for v, c in counts.items() if c == 3][0]
        kickers = sorted((v for v in vals if v != three), reverse=True)
        return (3, three, *kickers)
    if freq == [2, 2, 1]: # 兩對
        pairs = sorted((v for v, c in counts.items() if c == 2), reverse=True)
        kicker = [v for v in vals if counts[v] == 1][0]
        return (2, *pairs, kicker)
    if freq == [2, 1, 1, 1]: # 高牌
        pair = [v for v, c in counts.items() if c == 2][0]
        kickers = sorted((v for v in vals if v != pair), reverse=True)
        return (1, pair, *kickers)
    return (0, *vals)

def best_of_seven(hole, board):
    """從 7 張牌中選出最佳 5 張，回傳最大分數"""
    best = None
    for combo in itertools.combinations(hole + board, 5):
        score = hand_rank(combo)
        if best is None or score > best:
            best = score
    return best

# 3. 生成每種起手牌的所有具體花色組合
def get_specific_hole_combos(hand):
    r1, r2 = hand[0], hand[1]
    if len(hand) == 2:  # 對子
        return [[r1 + s1, r2 + s2] for i, s1 in enumerate(suits) for s2 in suits[i+1:]]
    if hand[2] == 's':  # 同花
        return [[r1 + s, r2 + s] for s in suits]
    if hand[2] == 'o':  # 不同花
        return [[r1 + s1, r2 + s2] for s1 in suits for s2 in suits if s1 != s2]

# 4. Monte Carlo 模擬（隨機花色、iters=2000）
def estimate_equity_montecarlo(hand, iters=2000):
    wins = ties = 0
    for _ in range(iters):
        # 隨機選一組具體花色的洞牌
        hole = random.choice(get_specific_hole_combos(hand))
        # 建立牌堆並移除洞牌
        deck = full_deck().copy()
        deck.remove(hole[0]); deck.remove(hole[1])
        # 隨機抽對手洞牌
        opp = random.sample(deck, 2)
        for c in opp: deck.remove(c)
        # 補滿 5 張公共牌
        board = random.sample(deck, 5)
        # 比較牌力
        my_score  = best_of_seven(hole, board)
        opp_score = best_of_seven(opp, board)
        if my_score > opp_score:
            wins += 1
        elif my_score == opp_score:
            ties += 1
    return wins / iters, ties / iters

# 5. 示範：估算 "AA" 的勝率
# win_rate, tie_rate = estimate_equity_montecarlo('AA', iters=2000)
# print(f"AA win_rate={win_rate:.3f}, tie_rate={tie_rate:.3f}")

# 6. 完整生成 CSV（離線執行）
combos = []
# i 對應第一張牌的 rank index，j 對應第二張
for i in range(len(ranks)):
    for j in range(len(ranks)):
        r1, r2 = ranks[i], ranks[j]
        if i < j:
            # 不同花 (offsuit)
            combos.append(r1 + r2 + 'o')
            # 同花 (suited)
            combos.append(r1 + r2 + 's')
        elif i == j:
            # 對子 (pair)
            combos.append(r1 + r2)
# 確保總共有 169 種
assert len(combos) == 169

results = []
count = 0
for hand in combos:
    print(f"Processing {hand} ({count+1}/{len(combos)})")
    count += 1
    w, t = estimate_equity_montecarlo(hand, iters=2000)
    results.append({'hand': hand, 'win_rate': w, 'tie_rate': t})
with open('preflop_equity_2000.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['hand','win_rate','tie_rate'])
    writer.writeheader()
    writer.writerows(results)
