from engine import *
from alphazero import *
import players as Player

# self.stateToInput is not implemented
# color counting in general is not implemented
# note: destination card points are only awarded at the end, not as they are completed
# longest route not implemented
# check mcts action hashing for children dictionary

if __name__ == "__main__":

    game_options = GameOptions(
        logs=False
    )
    alphazero_options = AlphaZeroTrainingOptions(
        game_options, 
        num_players=2
    )

    # ttr = TicketToRide(game_options)
    # ttr.play()

    ai = AlphaZeroTrainer(alphazero_options)
    ai.train()