import json
from game.game import setup_config, start_poker
# Import AI agents
from agents.call_player import setup_ai as call_ai
from agents.random_player import setup_ai as random_ai
from agents.console_player import setup_ai as console_ai
from agents.hybrid_player import setup_ai as hybrid_ai
# Baseline AI imports
from baseline0 import setup_ai as baseline0_ai
from baseline1 import setup_ai as baseline1_ai
from baseline2 import setup_ai as baseline2_ai
from baseline3 import setup_ai as baseline3_ai
from baseline4 import setup_ai as baseline4_ai # aggressive AI
from baseline5 import setup_ai as baseline5_ai
from baseline6 import setup_ai as baseline6_ai
from baseline7 import setup_ai as baseline7_ai

config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
config.register_player(name="myBot", algorithm=hybrid_ai())
config.register_player(name="baseline6", algorithm=baseline6_ai())

## Play in interactive mode if uncomment
#config.register_player(name="me", algorithm=console_ai())
game_result = start_poker(config, verbose=1)

print(json.dumps(game_result, indent=4))
