from numpy import ndarray
from players.ap_random import Random
from keras.api.layers import Dense, Input, BatchNormalization, ReLU, Add
from engine import GameEngine, GameOptions, Action
from keras.api.models import load_model, Sequential
from keras.api.losses import BinaryCrossentropy, CategoricalCrossentropy, MeanSquaredError
from keras.api.optimizers import Adam
from keras import Model

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
            # output = output[0]
            # self.action = output[0:4]
            # self.color_desire = output[4:(4+options.output_lengths[1])]
            # self.destination_desire_by_pickup_index = output[(4+options.output_lengths[1]):((4+options.output_lengths[1])+(options.output_lengths[2]))]
            # self.route_desire = output[((4+options.output_lengths[1])+(options.output_lengths[2])):(((4+options.output_lengths[1])+(options.output_lengths[2]))+(options.output_lengths[3]))]
            # self.win_chance = output[(((4+options.output_lengths[1])+(options.output_lengths[2]))+(options.output_lengths[3])):]
            self.action = output[0][0]
            self.color_desire = output[1][0]
            self.destination_desire_by_pickup_index = output[2][0]
            self.route_desire = output[3][0]
            self.win_chance = output[4][0]
            assert len(self.win_chance) == 1, f"[AutoTicketToRide] NeuralNetOutput: received output size {len(output)} but expecting {sum(options.output_lengths)} {options.output_lengths}"

    def __init__(self, options: NeuralNetOptions, load_from_path: str = None):
        assert isinstance(options, NeuralNetOptions)
        self.options = options
        self.model: Model = load_model(load_from_path) if load_from_path else self.new_model()

    def new_model(self):
        print(f"[AutoTicketToRide] NeuralNet: Creating new model for input size {self.options.state_size}")
        # model = Sequential()
        # model.add(Input((self.options.state_size,)))
        # model.add(Dense(511))
        # model.add(Dense(sum(self.options.output_lengths), name="output"))
        # model.compile(loss=BinaryCrossentropy())
        
        # model.add(Input((self.options.state_size,)))
        # model.add(Dense(800))
        # model.add(BatchNormalization())
        # model.add(ReLU())

        # for x in range(20):
        #     model.add(Dense(800))
        #     model.add(BatchNormalization())
        #     model.add(ReLU())
        #     model.add(Dense(800))
        #     model.add(BatchNormalization())
        #     model.add(Add([]))

        inputs = Input((self.options.state_size,))
        pre = Dense(800)(inputs)
        pre = BatchNormalization()(pre)
        pre = ReLU()(pre)

        for x in range(20):
            if x == 0: res_block = Dense(800)(pre)
            else: res_block = Dense(800)(res_block)
            res_block = BatchNormalization()(res_block)
            res_block = ReLU()(res_block)
            res_block = Dense(800)(res_block)
            res_block = BatchNormalization()(res_block)
            res_block = Add()([pre, res_block])
            res_block = ReLU()(res_block)
        
        action_out = Dense(400)(res_block)
        action_out = BatchNormalization()(action_out)
        action_out = ReLU()(action_out)
        action_out = Dense(4, name="action", activation="softmax")(action_out)

        color_out = Dense(400)(res_block)
        color_out = BatchNormalization()(color_out)
        color_out = ReLU()(color_out)
        color_out = Dense(9, name="color", activation="sigmoid")(color_out)

        dest_out = Dense(400)(res_block)
        dest_out = BatchNormalization()(dest_out)
        dest_out = ReLU()(dest_out)
        dest_out = Dense(self.options.output_lengths[2], name="destination", activation="sigmoid")(dest_out)

        route_out = Dense(400)(res_block)
        route_out = BatchNormalization()(route_out)
        route_out = ReLU()(route_out)
        route_out = Dense(self.options.output_lengths[3], name="route", activation="sigmoid")(route_out)

        win_out = Dense(400)(res_block)
        win_out = BatchNormalization()(win_out)
        win_out = ReLU()(win_out)
        win_out = Dense(1, name="win", activation="tanh")(win_out)
        
        model = Model(
            inputs=inputs,
            outputs=[action_out, color_out, dest_out, route_out, win_out]
        )

        losses = {
            "action": CategoricalCrossentropy(),
            "color": BinaryCrossentropy(),
            "destination": BinaryCrossentropy(),
            "route": BinaryCrossentropy(),
            "win": MeanSquaredError()
        }

        model.compile(Adam(), loss=losses)

        return model
    
    def inference(self, input: ndarray):
        return self.NeuralNetOutput(self.options, self.model.predict(input, verbose=0))
    
    def update_weights(self, inputs: list[int], outputs: dict[str, list]):
        self.model.fit(inputs, outputs, verbose=0)
    
    def save_to_file(self, filename: str):
        self.model.save(f"saved/{filename}.keras")
