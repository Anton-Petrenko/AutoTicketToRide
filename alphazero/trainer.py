import os
import math
from time import sleep
from ai import NeuralNet, NeuralNetOptions
from multiprocessing import Process, Queue, Manager
from engine import *

class AlphaZeroNode():
    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children: dict[Action, AlphaZeroNode] = {}
    
    def expanded(self):
        return len(self.children) > 0
    
    def value(self):
        if self.visit_count == 0: return 0
        return self.value_sum / self.visit_count

class AlphaZeroTrainingOptions():
    def __init__(
            self,
            game_options: GameOptions,
            num_players: int,
            max_moves_per_game: int = 300,
            max_games_stored: int = 15,
            path_network_to_load: str = None,
            gamma_noise_alpha: int = 0.1,
            root_exploration_fraction: int = 0.25,
            simulations_per_move: int = 800,
            pb_c_base: int = 19652,
            pb_c_init: int = 1.25,
            num_sampling_moves: int = 30,
            games_in_sampled_batch: int = 10,
            batch_size: int = 5
            ):
        game_options.players = None
        self.max_moves_per_game = max_moves_per_game
        self.game_options = game_options
        self.game_options.players = [Player(f"AlphaZero") for i in range(num_players)]
        self.network_path = path_network_to_load
        self.max_games_stored = max_games_stored
        self.gamma_noise_alpha = gamma_noise_alpha
        self.root_exploration_fraction = root_exploration_fraction
        self.simulations_per_move = simulations_per_move
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init
        self.num_sampling_moves = num_sampling_moves

        self.games_in_sampled_batch = games_in_sampled_batch
        self.batch_size = batch_size

class AlphaZeroTrainer():
    def __init__(self, options: AlphaZeroTrainingOptions):
        self.options = options

        game = GameEngine()
        game.setup_game(GameOptions(players=[Player("") for i in range(len(self.options.game_options.players))]))
        output_lengths = [4, 9, 0, 0, 1]
        output_lengths[2] = game.options.dests_dealt_on_request
        output_lengths[3] = len(game.get_routes())
        self.neural_net_options = NeuralNetOptions(len(self.options.game_options.players), len(game.state_representation()[0]), output_lengths)

    def train(self):

        if not os.path.exists("latest_network.keras"):
            NeuralNet(self.neural_net_options, self.options.network_path).model.save("latest_network.keras")

        # training_set_games = Queue(self.options.max_games_stored)
        manager = Manager()
        training_set_games = manager.list()
        
        game_gen_process = Process(target=self.generate_games, args=[training_set_games])
        network_update_process = Process(target=self.train_network, args=[training_set_games])

        network_update_process.start()
        game_gen_process.start()

        game_gen_process.join()
        network_update_process.join()

    def generate_games(self, training_set_games: list):
        network = NeuralNet(self.neural_net_options, "latest_network.keras")
        game = self.play_game(network)
        print(f"[{os.getpid()}] [AutoTicketToRide] AlphaZeroTrainer: Adding game to training set")
        training_set_games.append(game)

    def train_network(self, training_set_games: list[GameEngine]):
        network = NeuralNet(self.neural_net_options, "latest_network.keras")
        while len(training_set_games) != self.options.games_in_sampled_batch:
            print(f"[{os.getpid()}] [AutoTicketToRide] AlphaZeroTrainer: Training set has {len(training_set_games)} games")
            sleep(10)
        print(f"[{os.getpid()}] [AutoTicketToRide] AlphaZeroTrainer: Training set has {len(training_set_games)} games")
        batch = self.sample_batch(training_set_games)
        network.update_weights(batch)
    
    def sample_batch(self, training_set_games: list[GameEngine]) -> list[tuple[list[int], list[int]]]:
        move_sum = float(sum(len(game.history) for game in training_set_games))
        games = np.random.choice(
            training_set_games,
            size=self.options.batch_size,
            p=[len(game.history) / move_sum for game in training_set_games]
        )
        game_pos = [(game, np.random.randint(len(game.history))) for game in games]
        return [(game.history[index][1], self.make_target(game, game.history[index][1], game.history[index][0])) for (game, index) in game_pos]

    def make_target(self, game: GameEngine, state_representation: list[int], action: Action):
        assert isinstance(action, Action)

        action_target = [0, 0, 0, 0]
        if action.type == CHOOSE_DESTINATIONS: action_target[3] = 1
        else: action_target[action.type] = 1

        color_target = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        if action.type == PLACE_ROUTE:
            assert isinstance(action, PlaceRoute)
            for i, color in enumerate(action.color_precedence):
                color_target[COLOR_INDEXING[color]] = 1 - (0.1*i)
        elif action.type == DRAW_FACEUP:
            assert isinstance(action, DrawCard)
            color_target[COLOR_INDEXING[action.color]] = 1
        
        destination_target = [0]*self.neural_net_options.output_lengths[2]
        if action.type == CHOOSE_DESTINATIONS:
            assert isinstance(action, ChooseDestinations)
            assert action.faceup_destinations_at_time != None
            for i, destination in enumerate(action.faceup_destinations_at_time):
                for action_dest in action.destinations:
                    if action_dest.id == destination.id:
                        destination_target[i] = 1
        
        route_claim_target = [0]*len(game.get_routes())
        if action.type == PLACE_ROUTE:
            assert isinstance(action, PlaceRoute)
            route_claim_target[action.route.id] = 1
        
        game_win_prob_target = [0]
        if game.final_standings[0].turn_order == (game.turn-1)%len(game.options.players):
            game_win_prob_target = [1]
        
        return np.array(action_target + color_target + destination_target + route_claim_target + game_win_prob_target).reshape((1, 117))
        
    def play_game(self, network: NeuralNet):
        game = GameEngine()
        game.setup_game(self.options.game_options)
        while not game.game_ended and len(game.history) < self.options.max_moves_per_game:
            print(f"[{os.getpid()}] [AutoTicketToRide] AlphaZeroTrainer: Current game at turn {game.turn}")
            action, root = self.run_mcts(game, network)
            game.apply(action)
            # game.store_search_statistics(root)
        return game

    def run_mcts(self, game: GameEngine, network: NeuralNet):
        root = AlphaZeroNode(0)
        self.evaluate(root, game, network)
        self.add_exploration_noise(root)

        for _ in range(self.options.simulations_per_move):
            # print(f"[{os.getpid()}] [AutoTicketToRide] AlphaZeroTrainer: Simulating to find best move ... {_}/{self.options.simulations_per_move}")
            node = root
            game_clone = game.clone()
            search_path = [node]

            while node.expanded():
                action, node = self.select_child(node)
                game_clone.apply(action)
                search_path.append(node)
            
            value = self.evaluate(node, game_clone, network)
            self.backpropagate(search_path, value, game_clone.player_making_move)
        return self.select_action(game, root), root

    def select_action(self, game: GameEngine, root: AlphaZeroNode):
        visit_counts = [(child.visit_count, action) for action, child in root.children.items()]
        if len(game.history) < self.options.num_sampling_moves:
            visits, action = self.softmax_sample(visit_counts)
        else:
            visits, action = max(visit_counts)
        return action
    
    def softmax_sample(self, visit_counts: list[tuple[int, Action]]):
        total_visits = sum([visit_count[0] for visit_count in visit_counts])
        return max([(visit_count / total_visits, action) for visit_count, action in visit_counts])

    def backpropagate(self, search_path: list[AlphaZeroNode], value: float, to_play):
        for node in search_path:
            node.value_sum += value if node.to_play == to_play else (1 - value)
            node.visit_count += 1

    def select_child(self, node: AlphaZeroNode):
        score, action, child = max([(self.ucb_score(node, child), action, child) for action, child in node.children.items()], key=lambda a: a[0])
        return action, child

    def ucb_score(self, parent: AlphaZeroNode, child: AlphaZeroNode):
        pb_c = math.log((parent.visit_count + self.options.pb_c_base + 1) / self.options.pb_c_base) + self.options.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = child.value()
        return prior_score + value_score

    def add_exploration_noise(self, node: AlphaZeroNode):
        actions = node.children.keys()
        noise = np.random.gamma(self.options.gamma_noise_alpha, 1, len(actions))
        frac = self.options.root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

    def evaluate(self, node: AlphaZeroNode, game: GameEngine, network: NeuralNet):
        network_out = network.inference(game.state_representation())
        node.to_play = game.player_making_move
        if not game.game_ended:
            policy = {action: math.exp(self.get_logit_move(network_out, action, game.destinations_dealt)) for action in game.get_valid_moves()}
            policy_sum = sum(policy.values())
            for action, probability in policy.items():
                node.children[action] = AlphaZeroNode(probability / policy_sum)
        return network_out.win_chance
    
    def get_logit_move(self, network_out: NeuralNet.NeuralNetOutput, action: Action, destinations_dealt: list[DestinationCard]):
        assert len(network_out.action) == 4
        action_probability = network_out.action[0] if action.type == PLACE_ROUTE else network_out.action[1] if action.type == DRAW_FACEUP else network_out.action[2] if action.type == DRAW_FACEDOWN else network_out.action[3]
        color_desire_probability = 0.5
        dest_desire_probability = 0.5
        route_desire_probability = 0.5

        if action.type == PLACE_ROUTE:
            assert isinstance(action, PlaceRoute)
            for color in action.color_precedence:
                color_desire_probability += network_out.color_desire[COLOR_INDEXING[color]]
            color_desire_probability = color_desire_probability / len(action.color_precedence)
            route_desire_probability = network_out.route_desire[action.route.id]
        elif action.type  == DRAW_FACEUP:
            assert isinstance(action, DrawCard) and action.color
            color_desire_probability = network_out.color_desire[COLOR_INDEXING[action.color]]
        elif action.type == CHOOSE_DESTINATIONS:
            assert isinstance(action, ChooseDestinations)
            assert len(destinations_dealt) > 0
            for destination in action.destinations:
                dest_desire_probability += network_out.destination_desire_by_pickup_index[destinations_dealt.index(destination)]
            dest_desire_probability = dest_desire_probability / len(action.destinations)
        
        return (action_probability + color_desire_probability + dest_desire_probability + route_desire_probability) / 4