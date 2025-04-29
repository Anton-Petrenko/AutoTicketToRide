import keras
from copy import deepcopy
import numpy as np
from engine.lib import *
from engine.game import GameEngine, GameOptions
from ai.neuralnet import NeuralNet, NeuralNetOptions
from alphazero.trainer import AlphaZeroTrainer, AlphaZeroTrainingOptions
from players.ap_random import Random

class AlphaZero(Player):
    def __init__(self, name = "AlphaZero", filename=""):
        super().__init__(name)
        self.filename = filename
        self.model = NeuralNet(
            NeuralNetOptions(
                0,
                0,
                [0]
            ),
            self.filename
        )
        self.az = AlphaZeroTrainer(AlphaZeroTrainingOptions(GameOptions([Random(), Random()]), 2, simulations_per_move=500))

    def decide(self, game: GameEngine):
        possible_moves = game.get_valid_moves()
        if len(possible_moves) == 0: return None
        alt_game = GameEngine()
        alt_options = self.clone_game_options(game.options)
        self.copy_game_to(game, alt_game)
        alt_game.options = alt_options
        action, root = self.az.run_mcts(alt_game, self.model)
        return action
        # max_move = None
        # max_score = None
        # for move in possible_moves:
        #     score = self.get_logit_move(network_out, move, game.destinations_dealt)
        #     # print(move, score)
        #     if not max_move:
        #         max_move = move
        #         max_score = score
        #         continue
        #     if max_score < score:
        #         max_move = move
        #         max_score = score
        # return max_move

        # max_move = None
        # max_score = None
        # for move in possible_moves:
        #     score = self.get_move_score(move, network_out)
        #     if not max_move: 
        #         max_move = move
        #         max_score = score
        #         continue
        #     if max_score < score:
        #         max_move = move
        #         max_score = score
        # return max_move

    def copy_game_to(self, original: GameEngine, new: GameEngine):
        new.turn = deepcopy(original.turn)
        new.former_action = deepcopy(original.former_action)
        new.player_making_move = deepcopy(original.player_making_move)
        new.last_round_turn = deepcopy(original.last_round_turn)
        new.final_standings = deepcopy(original.final_standings)
        new.no_valid_moves_inarow = deepcopy(original.no_valid_moves_inarow)
        new.game_isset = deepcopy(original.game_isset)
        new.game_ended = deepcopy(original.game_ended)
        new.last_round = deepcopy(original.last_round)
        new.destinations_dealt = deepcopy(original.destinations_dealt)
        new.board = deepcopy(original.board)
        new.traincolor_discard_deck = deepcopy(original.traincolor_discard_deck)
        new.traincolor_discard_deck.shuffle(1)
        new.traincolor_deck = deepcopy(original.traincolor_deck)
        new.traincolor_deck.shuffle(1)
        new.faceup_cards = deepcopy(original.faceup_cards)
        new.destination_deck = deepcopy(original.destination_deck)
        new.destination_deck.shuffle(1)
        new.initial_round = deepcopy(original.initial_round)
        new.logs = []
        new.history = []
        new.child_visits = []
        new.player_id_history = []

    def clone_game_options(self, options: GameOptions):
        new_options = GameOptions(players=[])
        for old_player in options.players:
            new_player = Random()
            new_player.color_counts = deepcopy(old_player.color_counts)
            new_player.destinations = deepcopy(old_player.destinations)
            new_player.points = deepcopy(old_player.points)
            new_player.train_colors = deepcopy(old_player.train_colors)
            new_player.trains_left = deepcopy(old_player.trains_left)
            new_player.turn_order = deepcopy(old_player.turn_order)
            new_player.name = deepcopy(old_player.name)
            new_options.players.append(new_player)
        
        new_options.is_copy = True
        new_options.logs = False
        new_options.seed = deepcopy(options.seed)
        new_options.filename_paths = deepcopy(options.filename_paths)
        new_options.filename_dests = deepcopy(options.filename_dests)
        new_options.reshuffle_limit = deepcopy(options.reshuffle_limit)

        new_options.pink_trains = deepcopy(options.pink_trains)
        new_options.white_trains = deepcopy(options.white_trains)
        new_options.blue_trains = deepcopy(options.blue_trains)
        new_options.yellow_trains = deepcopy(options.yellow_trains)
        new_options.orange_trains = deepcopy(options.orange_trains)
        new_options.black_trains = deepcopy(options.black_trains)
        new_options.red_trains = deepcopy(options.red_trains)
        new_options.green_trains = deepcopy(options.green_trains)
        new_options.wild_trains = deepcopy(options.wild_trains)
        new_options.traincars_per_player = deepcopy(options.traincars_per_player)

        new_options.dests_dealt_per_player_start = deepcopy(options.dests_dealt_per_player_start)
        new_options.dests_dealt_on_request = deepcopy(options.dests_dealt_on_request)
        new_options.traincolor_dealt_per_player_start = deepcopy(options.traincolor_dealt_per_player_start)
        new_options.longest_route_bonus = deepcopy(options.longest_route_bonus)
        return new_options

    def get_move_score(self, action: Action, network_out: NeuralNet.NeuralNetOutput):
        total_score = 0
        if action.type == PLACE_ROUTE:
            assert isinstance(action, PlaceRoute)
            total_score = network_out.action[PLACE_ROUTE]*100
            total_score += network_out.route_desire[action.route.id]*10
            for i, color in enumerate(action.color_precedence):
                target = 0
                difference = abs(network_out.color_desire[COLOR_INDEXING[color]] - target)
                total_score += (5 - difference)
        elif action.type == DRAW_FACEUP:
            assert isinstance(action, DrawCard)
            total_score = network_out.action[DRAW_FACEUP]*100
            total_score += network_out.color_desire[COLOR_INDEXING[action.color]]
        elif action.type == DRAW_FACEDOWN:
            total_score = network_out.action[DRAW_FACEDOWN]*100
        elif action.type == DRAW_DESTINATIONS or action.type == CHOOSE_DESTINATIONS:
            total_score = network_out.action[DRAW_DESTINATIONS]*100
            if action.type == CHOOSE_DESTINATIONS:
                assert isinstance(action, ChooseDestinations)
                average = 0
                total = 0
                for i, dest in enumerate(action.faceup_destinations_at_time):
                    for action_dest in action.destinations:
                        if action_dest.id == dest.id:
                            average += network_out.destination_desire_by_pickup_index[i]
                            total += 1
                total_score += (average / total)
        else:
            raise TypeError()
        return total_score
    
    def get_logit_move(self, network_out: NeuralNet.NeuralNetOutput, action: Action, destinations_dealt: list[DestinationCard]):
        assert len(network_out.action) == 4
        action_probability = network_out.action[0] if action.type == PLACE_ROUTE else network_out.action[1] if action.type == DRAW_FACEUP else network_out.action[2] if action.type == DRAW_FACEDOWN else network_out.action[3]
        color_desire_probability = 1
        dest_desire_probability = 1
        route_desire_probability = 1

        if action.type == PLACE_ROUTE:
            assert isinstance(action, PlaceRoute)
            if action.route.color == "GRAY":
                sum_val = 0
                num_vals = len(action.color_precedence)
                for color in action.color_precedence:
                    sum_val += network_out.color_desire[COLOR_INDEXING[color]]
                color_desire_probability = sum_val / num_vals
        elif action.type  == DRAW_FACEUP:
            assert isinstance(action, DrawCard) and action.color
            color_desire_probability = network_out.color_desire[COLOR_INDEXING[action.color]]
        elif action.type == CHOOSE_DESTINATIONS:
            assert isinstance(action, ChooseDestinations)
            assert len(destinations_dealt) > 0
            for destination in action.destinations:
                dest_desire_probability += network_out.destination_desire_by_pickup_index[destinations_dealt.index(destination)]
            dest_desire_probability = dest_desire_probability / len(action.destinations)
        
        return action_probability * color_desire_probability * dest_desire_probability * route_desire_probability