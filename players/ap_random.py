from random import choice
from engine.lib import Player
from engine.game import GameEngine

class Random(Player):

    def __init__(self, name = "RandomAgent"):
        super().__init__(name)
        
    def decide(self, game: GameEngine):
        valid_moves = game.get_valid_moves()
        if len(valid_moves) == 0: return None
        else: return choice(valid_moves)