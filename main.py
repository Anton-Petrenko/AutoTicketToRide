from ai import *
from engine import *
import agents as Agents

# self.stateToInput is not implemented
# color counting in general is not implemented
# note: destination card points are only awarded at the end, not as they are completed
# longest route not implemented
# check mcts action hashing for children dictionary

if __name__ == "__main__":

    options = GameOptions(
        players=[Agents.RandomAgent(), Agents.FlatWinsMonteCarlo()]
    )

    ttr = TicketToRide(options)
    ttr.setup_game()
    ttr.play()