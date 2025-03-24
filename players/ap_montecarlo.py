from engine.lib import Player
from engine.game import GameEngine
from ai.montecarlo import FlatMonteCarlo, WINS

class FlatWinsMonteCarlo(Player):
    def __init__(self, name = "FlatWinsMonteCarlo", seconds_per_branch: int = 2):
        super().__init__(name)
        self.seconds_per_branch = seconds_per_branch
    
    def decide(self, game: GameEngine):
        valid_moves = game.get_valid_moves()
        if len(valid_moves) == 0: return None
        mcts = FlatMonteCarlo(game, self.seconds_per_branch, WINS)
        return mcts.find_action()

class ProgressivePruningBiasMonteCarlo(Player):
    def __init__(self, name = "PUB-MonteCarlo", seconds_per_branch: int = 2):
        super().__init__(name)
        self.seconds_per_branch = seconds_per_branch
    
    def decide(self, game: GameEngine):
        pass