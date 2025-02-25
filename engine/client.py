from .lib import *
from .game import *
from copy import deepcopy

class TicketToRide():

    def __init__(self, options: GameOptions):
        assert isinstance(options, GameOptions)
        self.options = options
        self.players_list = deepcopy(self.options.players)
        self.game_engine = None

    def setup_game(self):
        self.game_engine = GameEngine()
        self.game_engine.setup_game(self.options)

    def take_turns(self, turns: int = 1):
        assert getattr(self, 'game_engine') != None
        assert self.game_engine.game_isset == True
        assert self.game_engine.game_ended == False

        starting_turn = self.game_engine.turn
        while self.game_engine.turn != (starting_turn + turns):
            if self.game_engine.game_ended: break
            move = self.game_engine.options.players[self.game_engine.player_making_move].decide(self.game_engine)
            self.game_engine.apply(move)

    def play(self, num_games: int = 1):
        assert num_games > 0

        scores = [0]*len(self.options.players)
        if num_games == 1:
            assert getattr(self, 'game_engine') != None
            assert self.game_engine.game_isset == True
            assert self.game_engine.game_ended == False
            while not self.game_engine.game_ended:
                self.take_turns()
            for player in self.game_engine.final_standings:
                    scores[player.turn_order] += player.points

        else:
            for _ in range(num_games):
                print(f"Playing game {_+1}...")
                self.options.players = deepcopy(self.players_list)
                self.options.logs = False
                self.setup_game()
                while not self.game_engine.game_ended:
                    self.take_turns()
                for player in self.game_engine.final_standings:
                    scores[player.turn_order] += player.points
            
            print(f"Average scores across {num_games} games:")

        for i, score in enumerate(scores):
            print(f"[{GRAPH_COLORMAP[i]}] (Player {i+1}) {self.players_list[i].name} | {score/num_games}")