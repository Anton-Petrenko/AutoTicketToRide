import os
print(f"[{os.getpid()}] [AutoTicketToRide] main: Expect 2 additional TensorFlow import statements...")
from engine import *
from alphazero import *
import players as Player

# Check game win prob target in make_target - not 100% certain the player making move at that point + winner of game is being correctly deduced
# update_weights is very barebones right now compared to alphagozero pseudo.
# in neuralnet.py, take a look at the loss given to the compile() of the model... and the optimizer
# make_target() reshaping was solely for test purposes, be careful, the first number in shape is meant to be for the batch size in keras. do batch training instead of in the loop please.
# might be able to make things more efficient by weeding out deepcopy's of game variables which are not used in game copies...
# The optimizer is missing settings from alphagozero

if __name__ == "__main__":

    game_options = GameOptions(
        players=[Player.Random("Random1"), Player.Random("Random2")],
        logs=False,
        filename_paths="CT_paths.txt",
        filename_dests="CT_destinations.txt",
        red_trains=6,
        blue_trains=6,
        pink_trains=0,
        wild_trains=6,
        black_trains=0,
        green_trains=6,
        white_trains=0,
        orange_trains=0,
        yellow_trains=6,
        traincars_per_player=10
    )

    alphazero_options = AlphaZeroTrainingOptions(
        game_options,
        num_players=2,
        simulations_per_move=100,
        games_in_sampled_batch=50,
        num_sampling_moves=5,
        batch_size=1
    )

    # ttr = TicketToRide(game_options)
    # ttr.play(1)

    ai = AlphaZeroTrainer(alphazero_options)
    ai.train()