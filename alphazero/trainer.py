import os
from ai import NeuralNet, NeuralNetOptions
from multiprocessing import Process, Queue
from engine import GameEngine, GameOptions, Player

class AlphaZeroNode():
    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}

class AlphaZeroTrainingOptions():
    def __init__(
            self,
            game_options: GameOptions,
            num_players: int,
            max_moves_per_game: int = 300,
            max_games_stored: int = 15,
            path_network_to_load: str = None
            ):
        game_options.players = None
        self.max_moves_per_game = max_moves_per_game
        self.game_options = game_options
        self.game_options.players = [Player(f"AlphaZero") for i in range(num_players)]
        self.network_path = path_network_to_load
        self.max_games_stored = max_games_stored

class AlphaZeroTrainer():
    def __init__(self, options: AlphaZeroTrainingOptions):
        self.options = options
        self.neural_net_options = NeuralNetOptions(len(self.options.game_options.players))

    def train(self):

        if not os.path.exists("latest_network.keras"):
            NeuralNet(self.neural_net_options, self.options.network_path).model.save("latest_network.keras")

        training_set_games = Queue(self.options.max_games_stored)
        
        game_gen_process = Process(target=self.generate_games)
        network_update_process = Process(target=self.update_weights)

        game_gen_process.start()
        network_update_process.start()

        game_gen_process.join()
        network_update_process.join()

    def generate_games(self):
        network = NeuralNet(self.neural_net_options, "latest_network.keras")
        game = self.play_game(network)

    def update_weights(self):
        pass

    def play_game(self, network: NeuralNet):
        game = GameEngine()
        game.setup_game(self.options.game_options)
        while not game.game_ended and len(game.state_history) < self.options.max_moves_per_game:
            action, root = self.run_mcts(game, network)
            break
    
    def run_mcts(self, game: GameEngine, network: NeuralNet):
        # everything is set up to pick up where you left off here.
        root = AlphaZeroNode(0)
        self.evaluate(root, game, network)
    
    def evaluate(self, root: AlphaZeroNode, game: GameEngine, network: NeuralNet):
        print()