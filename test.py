import json
import time
from game.game import setup_config, start_poker
from agents.rule_base_player import setup_ai as my_agent
from agents.random_player import setup_ai as random_ai

# Import available baselines
try:
    from baseline0 import setup_ai as baseline0_ai
    from baseline1 import setup_ai as baseline1_ai
    from baseline2 import setup_ai as baseline2_ai
    from baseline3 import setup_ai as baseline3_ai
    from baseline4 import setup_ai as baseline4_ai
    from baseline5 import setup_ai as baseline5_ai
    from baseline6 import setup_ai as baseline6_ai
    from baseline7 import setup_ai as baseline7_ai
    baselines = [
        ("baseline0", baseline0_ai),
        ("baseline1", baseline1_ai),
        ("baseline2", baseline2_ai),
        ("baseline3", baseline3_ai),
        ("baseline4", baseline4_ai),
        ("baseline5", baseline5_ai),
        ("baseline6", baseline6_ai),
        ("baseline7", baseline7_ai),
    ]
except ImportError as e:
    print(f"Warning: Not all baselines could be imported: {e}")
    baselines = []
    for i in range(8):  # Try to import baselines 0-7
        try:
            baseline = __import__(f'baseline{i}', fromlist=['setup_ai'])
            baselines.append((f"baseline{i}", baseline.setup_ai))
        except ImportError:
            print(f"Could not import baseline{i}")

def evaluate_against_opponent(opponent_name, opponent_ai, num_games=50):
    """Evaluate our agent against a specific opponent"""
    print(f"Evaluating against {opponent_name}...")
    
    wins = 0
    total_stack = 0
    opponent_wins = 0
    opponent_stack = 0
    
    for game_idx in range(num_games):
        # Run game with our agent as player 1
        config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
        config.register_player(name="my_agent", algorithm=my_agent())
        config.register_player(name=opponent_name, algorithm=opponent_ai())
        game_result = start_poker(config, verbose=0)
        
        # Check who won
        players = game_result['players']
        my_result = next((p for p in players if p['name'] == "my_agent"), None)
        opp_result = next((p for p in players if p['name'] == opponent_name), None)
        
        if my_result['stack'] > opp_result['stack']:
            wins += 1
        elif opp_result['stack'] > my_result['stack']:
            opponent_wins += 1
            
        total_stack += my_result['stack']
        opponent_stack += opp_result['stack']
        
        # Print progress
        if (game_idx + 1) % 10 == 0:
            print(f"Completed {game_idx + 1}/{num_games} games")
    
    # Calculate win rate
    win_rate = wins / num_games * 100
    draw_rate = (num_games - wins - opponent_wins) / num_games * 100
    avg_stack = total_stack / num_games
    opponent_avg_stack = opponent_stack / num_games
    
    print(f"Results against {opponent_name}:")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Draw rate: {draw_rate:.2f}%")
    print(f"Average final stack: {avg_stack:.2f}")
    print(f"Opponent average final stack: {opponent_avg_stack:.2f}")
    print("-" * 50)
    
    return {
        "opponent": opponent_name,
        "win_rate": win_rate,
        "draw_rate": draw_rate,
        "avg_stack": avg_stack,
        "opponent_avg_stack": opponent_avg_stack
    }

def main():
    print("Starting evaluation of our poker agent...")
    start_time = time.time()
    
    results = []
    
    # Test against random player first for sanity check
    results.append(evaluate_against_opponent("random_player", random_ai, num_games=10))
    
    # Then test against all available baselines
    for name, ai_func in baselines:
        results.append(evaluate_against_opponent(name, ai_func, num_games=10))
    
    # Print summary
    print("\nEvaluation Summary:")
    print("=" * 60)
    print(f"{'Opponent':<15} {'Win Rate':<10} {'Draw Rate':<10} {'Avg Stack':<10} {'Opp Stack':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['opponent']:<15} {result['win_rate']:<10.2f}% {result['draw_rate']:<10.2f}% "
              f"{result['avg_stack']:<10.2f} {result['opponent_avg_stack']:<10.2f}")
    
    # Calculate overall stats
    overall_win_rate = sum(r['win_rate'] for r in results) / len(results)
    overall_avg_stack = sum(r['avg_stack'] for r in results) / len(results)
    
    print("=" * 60)
    print(f"Overall win rate: {overall_win_rate:.2f}%")
    print(f"Overall average stack: {overall_avg_stack:.2f}")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal evaluation time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()