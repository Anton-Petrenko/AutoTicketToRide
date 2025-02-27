from players.ap_random import Random
from keras.layers import Dense, Input
from engine import GameEngine, GameOptions
from keras.models import load_model, Sequential, Model

class NeuralNetOptions:
    def __init__(self, 
            num_players: int
        ):
        self.num_players = num_players

class NeuralNet:
    def __init__(self, options: NeuralNetOptions, load_from_path: str = None):
        assert isinstance(options, NeuralNetOptions)
        self.model: Model = load_model(load_from_path) if load_from_path else self.new_model(options.num_players)

    def new_model(self, num_players: int):
        state_size = GameEngine()
        state_size.setup_game(GameOptions(players=[Random() for i in range(num_players)]))
        state_size = len(state_size.state_representation())
        print(f"[AutoTicketToRide] NeuralNet: Creating new model for input size {state_size}")
        return Sequential([
            Input((state_size,)),
            Dense(511)
        ])
    
    def inference(self):
        pass