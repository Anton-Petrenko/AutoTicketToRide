import numpy as np
from collections import deque
from random import Random, randint

PLACE_ROUTE = 0
DRAW_FACEUP = 1
DRAW_FACEDOWN = 2
DRAW_DESTINATIONS = 3
CHOOSE_DESTINATIONS = 4
POINTS_BY_LENGTH = {1:1, 2:2, 3:4, 4:7, 5:10, 6:15}
COLOR_INDEXING = {'PINK': 0, 'WHITE': 1, 'BLUE': 2, 'YELLOW': 3, 'ORANGE': 4, 'BLACK': 5, 'RED': 6, 'GREEN': 7, 'WILD': 8}
GRAPH_COLORMAP = {0: 'blue', 1: 'red', 2: 'green', 3: 'orange', 4: 'black'}
CITY_LOCATIONS = {
    'VANCOUVER': np.array([-0.91, 0.90]), 
    'CALGARY': np.array([-0.62, 0.97]),
    'SEATTLE': np.array([-0.91, 0.69]),
    'HELENA': np.array([-0.39, 0.45]),
    'WINNIPEG': np.array([-0.11, 0.95]),
    'PORTLAND': np.array([-0.96, 0.49]),
    'DULUTH': np.array([0.13, 0.48]),
    'OMAHA': np.array([0.07, 0.11]),
    'DENVER': np.array([-0.26, -0.16]),
    'SALT LAKE CITY': np.array([-0.56, -0.03]),
    'SAULT ST. MARIE': np.array([0.41, 0.74]),
    'SAN FRANCISCO': np.array([-1, -0.28]),
    'TORONTO': np.array([0.66, 0.66]),
    'CHICAGO': np.array([0.4, 0.24]),
    'KANSAS CITY': np.array([0.11, -0.08]),
    'OKLAHOMA CITY': np.array([0.06, -0.42]),
    'SANTA FE': np.array([-0.28, -0.5]),
    'PHOENIX': np.array([-0.55, -0.71]),
    'LAS VEGAS': np.array([-0.68, -0.46]),
    'MONTREAL': np.array([0.84, 1]),
    'LOS ANGELES': np.array([-0.84, -0.69]),
    'PITTSBURGH': np.array([0.70, 0.30]),
    'SAINT LOUIS': np.array([0.3, -0.08]),
    'LITTLE ROCK': np.array([0.27, -0.44]),
    'DALLAS': np.array([0.11, -0.75]),
    'EL PASO': np.array([-0.29, -0.86]),
    'BOSTON': np.array([1, 0.76]),
    'NEW YORK': np.array([0.88, 0.46]),
    'WASHINGTON': np.array([0.9, 0.12]),
    'RALEIGH': np.array([0.78, -0.14]),
    'NASHVILLE': np.array([0.51, -0.24]),
    'NEW ORLEANS': np.array([0.42, -0.89]),
    'HOUSTON': np.array([0.2, -0.92]),
    'CHARLESTON': np.array([0.83, -0.4]),
    'ATLANTA': np.array([0.63, -0.37]),
    'MIAMI': np.array([0.91, -1])
}

class Player:

    def __init__(self, name):
        self.points = 0
        self.name = name
        self.turn_order = None
        self.trains_left = 45
        self.train_colors: list[str] = []
        self.destinations: list[DestinationCard] = []
    
    def decide(self):
        raise NotImplementedError()

class Deck:

    def __init__(self, items: list = [], seed: int = randint(0, 99999999)) -> None:
        self.rand = Random(seed)
        self.rand.shuffle(items)
        self.cards = deque(items)
    
    def draw(self, num: int) -> list:
        assert len(self.cards) >= num
        ret = []
        for _ in range(num):
            ret.append(self.cards.pop())
        return ret
    
    def insert(self, cards: list):
        assert type(cards) == list
        for card in cards:
            self.cards.appendleft(card)
    
    def shuffle(self, seed: int):
        if len(self.cards) == 0:
            return
        shuffled = list(self.cards)
        rand = Random(seed)
        rand.shuffle(shuffled)
        self.cards = deque(shuffled)

class Route:

    def __init__(self, city1: str, city2: str, weight: int, color: str, index: int):
        self.id: int = index
        self.city1: str = city1
        self.city2: str = city2
        self.color: str = color
        self.weight: int = weight

    def __str__(self) -> str:
        return f"ROUTE: [{self.id}] {self.city1} --{self.weight} {self.color}-- {self.city2}"

class DestinationCard:

    def __init__(self, city1: str, city2: str, points: int, index: int) -> None:
        self.id: int = index
        self.city1: str = city1
        self.city2: str = city2
        self.points: int = points
    
    def __str__(self) -> str:
        return f"DESTINATION: [{self.id}] {self.city1} --{self.points}-- {self.city2}"

class Action:
    def __init__(self, type: int, faceup_card_at_time: list[str]):
        assert isinstance(type, int)
        self.type = type
        self.faceup_card_at_time = faceup_card_at_time

class PlaceRoute(Action):
    def __init__(self, type: int, route: Route, color_precedence: list[str], faceup_card_at_time: list[str]):
        super().__init__(type, faceup_card_at_time)
        assert isinstance(route, Route)
        self.route: Route = route
        self.color_precedence: list[str] = color_precedence
    
    def __str__(self):
        return f"ACTION: Place route [{self.route.id}] {self.route.city1} --{self.route.weight} {self.route.color}-- {self.route.city2} using {self.color_precedence}"

class DrawCard(Action):
    def __init__(self, type: int, faceup_card_at_time: list[str], color: str = None):
        super().__init__(type, faceup_card_at_time)
        assert isinstance(color, str) if type == DRAW_FACEUP else True
        assert color == None if type == DRAW_FACEDOWN else True
        self.color: str | None = color

    def __str__(self):
        if self.color:
            return f"ACTION: Draw {self.color} from face up cards"
        else:
            return f"ACTION: Draw card from face down deck"

class ChooseDestinations(Action):
    def __init__(self, type: int, faceup_card_at_time: list[str], destinations: list[DestinationCard] = None):
        super().__init__(type, faceup_card_at_time)
        assert isinstance(destinations, list) if type == CHOOSE_DESTINATIONS else True
        assert isinstance(destinations, None) if type == DRAW_DESTINATIONS else True
        self.destinations: list[DestinationCard] | None = destinations
    
    def __str__(self):
        return f"ACTION: Choose {len(self.destinations)} destinations"

class DrawDestinations(Action):
    def __init__(self, type, faceup_card_at_time):
        super().__init__(type, faceup_card_at_time)
    def __str__(self):
        return f"ACTION: Draw destination cards"