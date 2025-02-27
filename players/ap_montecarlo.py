import ai as AITools
from engine.lib import Player
from engine.game import GameEngine

class FlatWinsMonteCarlo(Player):
    def __init__(self, name = "FlatWinsMonteCarlo", seconds_per_branch: int = 2):
        super().__init__(name)
        self.seconds_per_branch = seconds_per_branch
    
    def decide(self, game: GameEngine):
        valid_moves = game.get_valid_moves()
        if len(valid_moves) == 0: return None
        mcts = AITools.FlatMonteCarlo(game, self.seconds_per_branch, AITools.WINS)
        return mcts.find_action()