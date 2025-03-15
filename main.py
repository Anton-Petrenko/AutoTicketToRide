import os
print(f"[{os.getpid()}] [AutoTicketToRide] main: Expect 2 additional TensorFlow import statements...")
from engine import *
from alphazero import *
import players as Player

# longest route not implemented
# check mcts & alphazero action hashing for children dictionary
# what is a good way to get the probability of making a move given the neural net output?
# actions have faceupcard data that was previously used in the hashing function for it - ensure this is still happening
# figure out a good root_dirichlet_alpha is (research) honestly just look at add_exploration_noise functoin in general
# figure out what pb_c_base and pb_c_init is in the ucb formula
# make sure game.history is saving the right stuff
# rename num_sampling_moves to something more intuitive - find out what it means
# figure out store_search_statistics
# figure out exactly what softmax_sample is
# in sample_batch, history is stored as (action, game representation AFTER action)
# In logit move calc, are we taking into account the difference in notation of choosing/drawing routes? doesnt seem like choosing routes is accounted for here...
# Maybe look over that crazy assertion line in make_target when choosing destinations
# Check game win prob target in make_target - correctly deducing which player made the move at that point in the game?
# update_weights is very barebones right now compared to alphagozero pseudo.

#STOPPED AT: figuring out netwok training, specifically sample_batch

if __name__ == "__main__":

    game_options = GameOptions(
        logs=False
    )
    alphazero_options = AlphaZeroTrainingOptions(
        game_options, 
        num_players=2,
        simulations_per_move=1
    )

    # ttr = TicketToRide(game_options)
    # ttr.play()

    ai = AlphaZeroTrainer(alphazero_options)
    ai.train()