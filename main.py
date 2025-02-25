from engine import *
from agents import *

# self.stateToInput is not implemented
# color counting in general is not implemented
# note: destination card points are only awarded at the end, not as they are completed
# longest route not implemented

options = GameOptions(
    players=[RandomAgent(), RandomAgent()]
)

ttr = TicketToRide(options)