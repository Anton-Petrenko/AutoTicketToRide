from numpy import ndarray
from players.ap_random import Random
from keras.layers import Dense, Input
from engine import GameEngine, GameOptions
from keras.models import load_model, Sequential, Model

class NeuralNetOptions:
    def __init__(
            self,
            num_players: int,
            state_size: int,
            output_lengths: list[int]
        ):
        self.num_players = num_players
        self.state_size = state_size
        self.output_lengths = output_lengths

class NeuralNet:

    class NeuralNetOutput:
        def __init__(self, options: NeuralNetOptions, output: ndarray):
            output = output[0]
            self.action = output[0:4]
            self.color_desire = output[4:(4+options.output_lengths[1])]
            self.destination_desire_by_pickup_index = output[(4+options.output_lengths[1]):((4+options.output_lengths[1])+(options.output_lengths[2]))]
            self.route_desire = output[((4+options.output_lengths[1])+(options.output_lengths[2])):(((4+options.output_lengths[1])+(options.output_lengths[2]))+(options.output_lengths[3]))]
            self.win_chance = output[(((4+options.output_lengths[1])+(options.output_lengths[2]))+(options.output_lengths[3])):]
            assert len(self.win_chance) == 1, f"[AutoTicketToRide] NeuralNetOutput: received output size {len(output)} but expecting {sum(options.output_lengths)} {options.output_lengths}"

    def __init__(self, options: NeuralNetOptions, load_from_path: str = None):
        assert isinstance(options, NeuralNetOptions)
        self.options = options
        self.model: Model = load_model(load_from_path) if load_from_path else self.new_model()

    def new_model(self):
        print(f"[AutoTicketToRide] NeuralNet: Creating new model for input size {self.options.state_size}")
        model = Sequential()
        model.add(Input((self.options.state_size,)))
        model.add(Dense(511))
        model.add(Dense(sum(self.options.output_lengths), name="output"))
        model.compile()
        return model
    
    def inference(self, input: ndarray):
        return self.NeuralNetOutput(self.options, self.model.predict(input, verbose=0))
    