import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from game.game import setup_config, start_poker
from agents.rule_base_player import setup_ai as my_agent
from agents.random_player import setup_ai as random_ai

# 載入 baselines（同原本程式）
baselines = []
for i in range(8):
    try:
        mod = __import__(f'baseline{i}', fromlist=['setup_ai'])
        baselines.append((f"baseline{i}", mod.setup_ai))
    except ImportError:
        pass

def simulate_one_game(opponent_ai):
    """模擬單一場對局，回傳 (my_stack, opp_stack)"""
    config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
    config.register_player(name="my_agent", algorithm=my_agent())
    config.register_player(name="opponent",   algorithm=opponent_ai())
    result = start_poker(config, verbose=0)
    p = result['players']
    my_stack  = next(ply['stack'] for ply in p if ply['name']=="my_agent")
    opp_stack = next(ply['stack'] for ply in p if ply['name']=="opponent")
    return my_stack, opp_stack

def evaluate_against_opponent(opponent_name, opponent_ai, num_games=50, workers=4):
    print(f"Evaluating against {opponent_name} with {workers} workers...")
    wins = opponent_wins = 0
    total_my = total_opp = 0

    # 用 ProcessPoolExecutor 平行運算
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # 提交所有模擬任務
        futures = [executor.submit(simulate_one_game, opponent_ai) for _ in range(num_games)]
        for idx, fut in enumerate(as_completed(futures), 1):
            my_stack, opp_stack = fut.result()
            total_my  += my_stack
            total_opp += opp_stack
            if my_stack > opp_stack:
                wins += 1
            elif opp_stack > my_stack:
                opponent_wins += 1
            # 進度
            if idx % 10 == 0 or idx == num_games:
                print(f"  Completed {idx}/{num_games}")

    win_rate      = wins / num_games * 100
    draw_rate     = (num_games - wins - opponent_wins) / num_games * 100
    avg_my_stack  = total_my  / num_games
    avg_opp_stack = total_opp / num_games

    print(f"Results against {opponent_name}: win {win_rate:.2f}%, draw {draw_rate:.2f}%")
    return {
        "opponent": opponent_name,
        "win_rate": win_rate,
        "draw_rate": draw_rate,
        "avg_stack": avg_my_stack,
        "opponent_avg_stack": avg_opp_stack
    }

def main():
    print("Starting parallel evaluation of our poker agent...")
    start_time = time.time()

    results = []
    # sanity check
    results.append(evaluate_against_opponent("random_player", random_ai, num_games=20, workers=4))
    # 各 baseline
    for name, ai in baselines:
        results.append(evaluate_against_opponent(name, ai, num_games=12, workers=4))

    # 最後列印總表（同原本程式）
    print("\nEvaluation Summary:")
    print(f"{'Opp':<12}{'Win%':>11}{'Draw%':>8}{'MyAvg':>10}{'OppAvg':>10}")
    for r in results:
        print(f"{r['opponent']:<13}{r['win_rate']:>8.2f}%{r['draw_rate']:>8.2f}%"
              f"{r['avg_stack']:>10.2f}{r['opponent_avg_stack']:>10.2f}")
    print(f"Overall win rate: {sum(r['win_rate'] for r in results)/len(results):.2f}%")
    print(f"Total time: {time.time()-start_time:.2f}s")

if __name__=="__main__":
    main()
