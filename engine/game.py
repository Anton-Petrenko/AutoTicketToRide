import re
import matplotlib
import matplotlib.pyplot
from .lib import *
import networkx as nx
from random import randint
from itertools import combinations

class GameOptions:
    def __init__(
            self,
            players: list[Player],
            logs: bool = True,
            seed: int = randint(0, 999999999),
            is_copy: bool = False,
            filename_paths: str = "USA_paths.txt",
            filename_dests: str = "USA_destinations.txt",
            reshuffle_limit: int = 10,
            dests_dealt_per_player_start: int = 3
            ):
        
        assert 2 <= len(players) <= 5
        self.players = players
        
        self.is_copy = is_copy
        self.logs = logs
        self.seed = seed
        self.filename_paths = filename_paths
        self.filename_dests = filename_dests
        self.reshuffle_limit = reshuffle_limit

        assert 0 < dests_dealt_per_player_start
        self.dests_dealt_per_player_start = 3

class GameEngine:

    def __init__(self):
        pass

    def setup_game(self, options: GameOptions):
        assert isinstance(options, GameOptions)

        self.turn = 0
        self.logs = []
        self.states = []
        self.is_copy = options.is_copy
        self.former_action = None
        self.player_making_move = 0
        self.last_round_turn = None
        self.final_standings = None
        self.no_valid_moves_inarow = 0

        self.options = options
        self.game_isset = True
        self.game_ended = False
        self.last_round = False
        self.destinations_dealt: list[DestinationCard] = []

        self.board: nx.MultiGraph = nx.MultiGraph()
        self.board.add_edges_from((route.city1, route.city2, {'weight': route.weight, 'color': route.color, 'owner': None, 'index': route.id}) for route in self.get_routes(options.filename_paths))
        
        self.traincolor_discard_deck: Deck = Deck()
        self.traincolor_deck: Deck[str] = self.get_traincolors()

        for player in options.players:
            player.train_colors.extend(self.traincolor_deck.draw(4))

        self.faceup_cards: list[str] = self.traincolor_deck.draw(5)
        self.validate_faceup_cards()

        self.destination_deck: Deck[DestinationCard] = Deck(self.get_destinations(options.filename_dests))
        
        for i, player in enumerate(options.players):
            player.turn_order = i

        self.initial_round = True
        self.destinations_dealt.extend(self.destination_deck.draw(options.dests_dealt_per_player_start))
    
    def get_routes(self, filename: str = "USA_paths.txt") -> list[Route]:

        filename = filename if filename.endswith(".txt") else f'{filename}.txt'
        
        ret = []
        lines = open(f"engine/datafiles/{filename}").readlines()
        for i, path in enumerate(lines):
            data = re.search('(^\D+)(\d)\W+(\w+)\W+(.+)', path)
            ret.append(Route(data.group(1).strip(), data.group(4).strip(), int(data.group(2).strip()), data.group(3).strip(), i))
        return ret
    
    def get_destinations(self, filename: str = "USA_destinations.txt") -> list[DestinationCard]:

        filename = filename if filename.endswith(".txt") else f'{filename}.txt'

        ret = []
        lines = open(f"engine/datafiles/{filename}").readlines()
        for i, path in enumerate(lines):
            data = re.search('(^\D+)(\d+)\s(.+)', path)
            ret.append(DestinationCard(data.group(1).strip(), data.group(3).strip(), int(data.group(2).strip()), i))
        return ret
    
    def get_traincolors(self, pink: int = 12, white: int = 12, blue: int = 12, yellow: int = 12, orange: int = 12, black: int = 12, red: int = 12, green: int = 12, wild: int = 12):
        
        deck = ['PINK']*pink+['WHITE']*white+['BLUE']*blue+['YELLOW']*yellow+['ORANGE']*orange+['BLACK']*black+['RED']*red+['GREEN']*green+['WILD']*wild
        return Deck(deck)
    
    def validate_faceup_cards(self):

        if self.faceup_cards.count('WILD') >= 3:
            self.traincolor_discard_deck.insert(self.faceup_cards)
            self.faceup_cards = []
            iter = 0
            while len(self.faceup_cards) <= 5 and iter != self.options.reshuffle_limit:
                try:
                    self.faceup_cards.extend(self.traincolor_deck.draw(1))
                except:
                    self.traincolor_discard_deck.shuffle(self.options.seed + iter)
                    self.traincolor_deck.insert(list(self.traincolor_discard_deck.cards))
                    self.traincolor_discard_deck = Deck()
                    iter += 1
        
        if len(self.traincolor_deck.cards) <= 2 and len(self.traincolor_discard_deck.cards) > 0:
            self.traincolor_discard_deck.shuffle(self.options.seed)
            self.traincolor_deck.insert(list(self.traincolor_discard_deck.cards))
            self.traincolor_discard_deck = Deck()
    
    def get_valid_moves(self):
        assert self.game_ended == False
        assert self.game_isset == True

        valid_moves: list[Action] = []

        if self.initial_round:
            x = len(self.destinations_dealt)
            helper = list(range(0, x))
            for y in range(2, x+1):
                for z in combinations(helper, y):
                    dests_to_take = []
                    for index in list(z):
                        dests_to_take.append(self.destinations_dealt[index])
                    valid_moves.append(ChooseDestinations(CHOOSE_DESTINATIONS, self.faceup_cards, dests_to_take))
            return valid_moves

        if self.former_action:
            if self.former_action.type == DRAW_FACEDOWN or self.former_action.type == DRAW_FACEUP:
                valid_moves.extend([DrawCard(DRAW_FACEUP, self.faceup_cards.copy(), color) for color in self.faceup_cards if color != 'WILD'])
                if len(self.traincolor_deck.cards) > 0:
                    valid_moves.append(DrawCard(DRAW_FACEDOWN, self.faceup_cards.copy()))
            elif self.former_action.type == DRAW_DESTINATIONS:
                x = len(self.destinations_dealt)
                helper = list(range(0, x))
                for y in range(1, x+1):
                    for z in combinations(helper, y):
                        dests_to_take = []
                        for index in list(z):
                            dests_to_take.append(self.destinations_dealt[index])
                        valid_moves.append(ChooseDestinations(CHOOSE_DESTINATIONS, self.faceup_cards, dests_to_take))
        else:
            valid_moves.extend(self.claimable_route_actions())
            valid_moves.extend([DrawCard(DRAW_FACEUP, self.faceup_cards.copy(), color) for color in self.faceup_cards])
            if len(self.traincolor_deck.cards) > 0:
                valid_moves.append(DrawCard(DRAW_FACEDOWN, self.faceup_cards.copy()))
            if len(self.destination_deck.cards) > 0:
                valid_moves.append(DrawDestinations(DRAW_DESTINATIONS, self.faceup_cards.copy()))
        
        return valid_moves
    
    def claimable_route_actions(self, player: Player = None) -> list[PlaceRoute]:
        player = self.options.players[self.player_making_move] if player == None else player
        assert isinstance(player, Player)

        ret = []
        for route in self.board.edges(data=True):

            if route[2]['owner'] != None:
                continue
            
            all_routes_between: dict = self.board[route[0]][route[1]]
            claimable = True
            if len(all_routes_between) == 2:
                if 2 <= len(self.options.players) <= 3:
                    for data in all_routes_between.values():
                        if data['owner'] != None:
                            claimable = False
                else:
                    for data in all_routes_between.values():
                        if data['owner'] == player.turn_order:
                            claimable = False
            
            if not claimable:
                continue
            
            if player.trains_left <= int(route[2]['weight']):
                continue
            
            player_wilds = player.train_colors.count("WILD")
            if player_wilds >= int(route[2]['weight']):
                ret.append(PlaceRoute(PLACE_ROUTE, Route(route[0], route[1], int(route[2]['weight']), route[2]['color'], int(route[2]['index'])), ['WILD'], self.faceup_cards.copy()))

            if route[2]['color'] != 'GRAY':
                num_colors_in_hand = player.train_colors.count(route[2]['color'])

                if num_colors_in_hand >= route[2]['weight']:
                    ret.append(PlaceRoute(PLACE_ROUTE, Route(route[0], route[1], int(route[2]['weight']), route[2]['color'], int(route[2]['index'])), [route[2]['color']], self.faceup_cards.copy()))

                if player_wilds > 0:
                    if player_wilds < route[2]['weight'] and num_colors_in_hand + player_wilds >= route[2]['weight']:
                        ret.append(PlaceRoute(PLACE_ROUTE, Route(route[0], route[1], int(route[2]['weight']), route[2]['color'], int(route[2]['index'])), ['WILD', route[2]['color']], self.faceup_cards.copy()))
                    if num_colors_in_hand < route[2]['weight'] and num_colors_in_hand + player_wilds >= route[2]['weight']:
                        ret.append(PlaceRoute(PLACE_ROUTE, Route(route[0], route[1], int(route[2]['weight']), route[2]['color'], int(route[2]['index'])), [route[2]['color'], 'WILD'], self.faceup_cards.copy()))
            else:
                for color in COLOR_INDEXING.keys():
                    num_colors_in_hand = player.train_colors.count(color)
                    if num_colors_in_hand == 0 or color == 'WILD': continue

                    if num_colors_in_hand >= route[2]['weight']:
                        ret.append(PlaceRoute(PLACE_ROUTE, Route(route[0], route[1], int(route[2]['weight']), route[2]['color'], int(route[2]['index'])), [color], self.faceup_cards.copy()))

                    if player_wilds > 0:
                        if player_wilds < route[2]['weight'] and num_colors_in_hand + player_wilds >= route[2]['weight']:
                            ret.append(PlaceRoute(PLACE_ROUTE, Route(route[0], route[1], int(route[2]['weight']), route[2]['color'], int(route[2]['index'])), ['WILD', color], self.faceup_cards.copy()))
                        if num_colors_in_hand < route[2]['weight'] and num_colors_in_hand + player_wilds >= route[2]['weight']:
                            ret.append(PlaceRoute(PLACE_ROUTE, Route(route[0], route[1], int(route[2]['weight']), route[2]['color'], int(route[2]['index'])), [color, 'WILD'], self.faceup_cards.copy()))

        return ret

    def choose_destinations(self, action: ChooseDestinations):
        assert isinstance(action, ChooseDestinations)
        assert len(self.destinations_dealt) > 0

        self.add_log_line(f"Picks {len(action.destinations)} destinations", 1)

        for destination in action.destinations:
            assert destination in self.destinations_dealt
            self.add_log_line(str(destination), 2)
            self.destinations_dealt.remove(destination)
            self.options.players[self.player_making_move].destinations.append(destination)
            self.options.players[self.player_making_move].points -= destination.points

    def end_turn(self):

        if self.initial_round:
            self.add_log_line("")
            self.destination_deck.insert(self.destinations_dealt)
            self.destinations_dealt = []
            self.player_making_move = (self.player_making_move + 1) % len(self.options.players)
            if len(self.options.players[self.player_making_move].destinations) == 0: 
                self.destinations_dealt.extend(self.destination_deck.draw(self.options.dests_dealt_per_player_start))
                return
            else:
                self.initial_round = False
                return

        if not self.former_action:
            self.add_log_line("")
            self.player_making_move = (self.player_making_move + 1) % len(self.options.players)
            self.destination_deck.insert(self.destinations_dealt)
            self.destinations_dealt = []
            self.turn += 1
        
        if not self.last_round:
            for player in self.options.players:
                if player.trains_left < 0:
                    raise ValueError()
                if player.trains_left < 3:
                    self.last_round = True
                    self.last_round_turn = self.turn
                    self.add_log_line("This turn has triggered the last round!", 1)
        else:
            if self.turn == self.last_round_turn + len(self.options.players):
                self.end_game()
                return

        self.validate_faceup_cards()

    def end_game(self):
        self.game_ended = True
        
        for player in self.options.players:
            self.add_log_line(f"Player {player.turn_order} - {player.name}:")
            player_board = nx.MultiGraph()
            player_board.add_edges_from([edge for edge in self.board.edges(data=True) if edge[2]['owner'] == player.turn_order])
            for claimed in player_board.edges(data=True):
                self.add_log_line(f"(+{POINTS_BY_LENGTH[claimed[2]['weight']]}) placed {claimed[0]} --{claimed[2]['weight']} {claimed[2]['color']}-- {claimed[1]}", 1)
            for destination in player.destinations:
                if player_board.has_node(destination.city1) and player_board.has_node(destination.city2):
                    if nx.has_path(player_board, destination.city1, destination.city2):
                        player.points += destination.points
                        self.add_log_line(f"(+{destination.points}) {destination} completed", 1)
                    else: self.add_log_line(f"(-{destination.points}) {destination} not completed", 1)
                else: self.add_log_line(f"(-{destination.points}) {destination} not completed", 1)
            
        self.add_log_line("")
        self.final_standings = sorted([player for player in self.options.players], key = lambda player: player.points, reverse=True)
        for i, player in enumerate(self.final_standings):
            self.add_log_line(f"{i+1}. (Player {player.turn_order}) {player.name} | {player.points} points")

    def apply(self, action: Action):

        if action == None:
            self.no_valid_moves_inarow += 1
            self.former_action = None
            self.end_turn()
            if self.no_valid_moves_inarow == len(self.options.players):
                self.end_game()
            return

        assert isinstance(action, Action)
        self.no_valid_moves_inarow = 0
        
        if self.former_action == None:
            if self.initial_round: self.add_log_line(f"PREGAME: Player {self.player_making_move} - {self.options.players[self.player_making_move].name}")
            else: self.add_log_line(f"TURN {self.turn}: Player {self.player_making_move} - {self.options.players[self.player_making_move].name}")
            self.add_log_line(f"")

        if action.type == CHOOSE_DESTINATIONS:
            assert isinstance(action, ChooseDestinations)
            self.choose_destinations(action)
            self.former_action = None
        elif action.type == DRAW_DESTINATIONS:
            assert isinstance(action, DrawDestinations)
            self.deal_destinations()
            self.former_action = action
        elif action.type == DRAW_FACEUP:
            assert isinstance(action, DrawCard)
            self.draw_faceup(action.color)
            self.former_action = None if action.color == 'WILD' or isinstance(self.former_action, DrawCard) else action
        elif action.type == DRAW_FACEDOWN:
            assert isinstance(action, DrawCard)
            self.draw_facedown()
            self.former_action = None if isinstance(self.former_action, DrawCard) else action
        elif action.type == PLACE_ROUTE:
            assert isinstance(action, PlaceRoute)
            self.place_route(action)
            self.former_action = None
        else:
            raise ValueError()
        
        self.end_turn()
    
    def place_route(self, action):
        assert isinstance(action, PlaceRoute)

        self.add_log_line(f"[PLAYER COLORS]: {self.options.players[self.player_making_move].train_colors}", 1)

        found = False
        for path in self.board[action.route.city1][action.route.city2].values():
            if path['index'] == action.route.id:
                found = True
                path['owner'] = self.player_making_move
        assert found

        colors_to_use = []
        for color in action.color_precedence:
            while color in self.options.players[self.player_making_move].train_colors and len(colors_to_use) < action.route.weight:
                self.options.players[self.player_making_move].train_colors.remove(color)
                colors_to_use.append(color)
        
        assert action.route.weight == len(colors_to_use)
        self.options.players[self.player_making_move].trains_left -= action.route.weight
        self.options.players[self.player_making_move].points += POINTS_BY_LENGTH[action.route.weight]
        self.traincolor_discard_deck.insert(colors_to_use)

        self.add_log_line(f"Placed {action.route}", 1)
        self.add_log_line(f"using {colors_to_use} derived from {action.color_precedence}", 2)
        self.add_log_line(f"[PLAYER COLORS]: {self.options.players[self.player_making_move].train_colors}", 1)

    def draw_facedown(self):
        card_drawn = self.traincolor_deck.draw(1)
        self.options.players[self.player_making_move].train_colors.extend(card_drawn)
        self.add_log_line(f"[FACEDOWN DECK]: Picked up {card_drawn[0]}", 1)

    def draw_faceup(self, color: str):
        assert color in self.faceup_cards

        self.add_log_line(f"[GAME FACEUP]: {self.faceup_cards}", 1)

        if self.former_action: 
            assert isinstance(self.former_action, DrawCard)
            assert color != 'WILD'
            self.faceup_cards.remove(color)
            if len(self.traincolor_deck.cards) > 0:
                self.faceup_cards.extend(self.traincolor_deck.draw(1))
            self.options.players[self.player_making_move].train_colors.append(color)
        else:
            self.faceup_cards.remove(color)
            if len(self.traincolor_deck.cards) > 0:
                self.faceup_cards.extend(self.traincolor_deck.draw(1))
            self.options.players[self.player_making_move].train_colors.append(color)
        
        self.add_log_line(f"[FACEUP DECK]: Picked up {color}", 1)

    def deal_destinations(self):

        while len(self.destinations_dealt) < 3 and len(self.destination_deck.cards) > 0:
            self.destinations_dealt.extend(self.destination_deck.draw(1))
        
        assert len(self.destinations_dealt) > 0
        self.add_log_line("Dealing destination cards:", 1)
        for card in self.destinations_dealt:
            self.add_log_line(str(card), 2)

    def add_log_line(self, log: str, indent: int = 0):
        if not self.options.logs: return
        if not log.endswith("\n"): log = f"{log}\n"
        line = ["\t"]*indent
        line.append(log)
        self.logs.append("".join(line))
    
    def save_log(self, filename: str = "log.txt"):
        file = open(filename, "w")
        file.writelines(self.logs)
        file.close()
    
    def visualize_board(self):
        if self.options.filename_dests == "USA_destinations.txt": pos = CITY_LOCATIONS
        else: pos = nx.spectral_layout(self.board)
        nx.draw_networkx_nodes(self.board, pos)
        nx.draw_networkx_labels(self.board, pos, font_size = 6)
        for player in self.options.players:
            edges = [edge for edge in self.board.edges(data=True) if edge[2]['owner'] == player.turn_order]
            nx.draw_networkx_edges(self.board, pos, edges, edge_color=GRAPH_COLORMAP[player.turn_order], connectionstyle=f"arc3, rad = 0.{player.turn_order}", arrows=True)
        matplotlib.pyplot.title(f"AutoTTR Engine Game\nTurn {self.turn}")
        matplotlib.pyplot.show()