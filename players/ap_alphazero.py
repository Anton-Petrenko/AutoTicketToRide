import numpy as np
from engine.lib import *
from engine.game import GameEngine
from ai.neuralnet import NeuralNet, NeuralNetOptions

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
    def decide(self, game: GameEngine):
        possible_moves = game.get_valid_moves()
        if len(possible_moves) == 0: return None

        highest_logit = None
        action = None
        for move in possible_moves:
            network_out = self.model.inference(np.array([game.state_representation()]))
            logit = self.logit(network_out, move, game.destinations_dealt)
            if highest_logit == None or logit > highest_logit:
                highest_logit = logit
                action = move
        return action

    def logit(self, network_out: NeuralNet.NeuralNetOutput, action: Action, destinations_dealt: list[DestinationCard]):
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