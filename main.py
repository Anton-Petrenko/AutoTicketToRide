from engine import GameOptions, TicketToRide
# from alphazero import *
import players as Player

# Check game win prob target in make_target - not 100% certain the player making move at that point + winner of game is being correctly deduced
# update_weights is very barebones right now compared to alphagozero pseudo.
# in neuralnet.py, take a look at the loss given to the compile() of the model... and the optimizer
# make_target() reshaping was solely for test purposes, be careful, the first number in shape is meant to be for the batch size in keras. do batch training instead of in the loop please.
# might be able to make things more efficient by weeding out deepcopy's of game variables which are not used in game copies...
# The optimizer is missing settings from alphagozero
# Note: in montecarloPUB, deciphering which routes are on the "path" is currently being done by finding the shortest way. However, there are ways that may be worth more points, but not the shortest.
# Do we all the points earned in the game to the nodes "total_score" on that path in montecarloPUB?
# montecarlopub does not have progressive unpruning
# montecarlopub free routes graph is good for CT map, but needs the extra conditionals for USA map
# in longroutejunkie check if the destinations_not_completed function is doing the card.city1 IN correctly. (and card.city2 IN)

if __name__ == "__main__":

    game_options = GameOptions(
        players=[Player.OneStep(), Player.LongRoute(), Player.Path(), Player.Hungry()],
        logs=True,
        filename_paths="USA_paths.txt",
        filename_dests="USA_destinations.txt",
        # red_trains=6,
        # blue_trains=6,
        # pink_trains=0,
        # wild_trains=6,
        # black_trains=0,
        # green_trains=6,
        # white_trains=0,
        # orange_trains=0,
        # yellow_trains=6,
        # traincars_per_player=10,
        longest_route_bonus=True
    )

    # alphazero_options = AlphaZeroTrainingOptions(
    #     game_options,
    #     num_players=2,
    #     simulations_per_move=100,
    #     games_in_sampled_batch=50,
    #     num_sampling_moves=5,
    #     batch_size=1
    # )

    ttr = TicketToRide(game_options)
    ttr.play()
    # hungry = Player.Hungry()
    # hungry.decide(ttr.game_engine)
    # ttr.play()
    # ttr.game_engine.visualize_board()

    # ai = AlphaZeroTrainer(alphazero_options)
    # ai.train()