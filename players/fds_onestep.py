import networkx as nx
from engine.lib import *
from engine.game import GameEngine

class OneStep(Player):

    def __init__(self, name = "OneStepAgent"):
        super().__init__(name)
        self.current_objective_route = None
        self.current_objective_color = None
        self.players_previous_points = -1
    
    def decide(self, game: GameEngine):
        possible_moves = game.get_valid_moves()
        if len(possible_moves) == 0:
            return None
        
        if game.initial_round:
            for m in possible_moves:
                assert isinstance(m, ChooseDestinations)
                if len(m.destinations) == game.options.dests_dealt_per_player_start:
                    return m
        
        elif possible_moves[0].type == CHOOSE_DESTINATIONS:
            self.players_previous_points = -1
            return self.chooseDestinations(possible_moves, game)
        
        claim_route_moves: list[PlaceRoute] = []
        draw_train_card_moves: list[DrawCard] = []

        for move in possible_moves:
            if move.type == DRAW_DESTINATIONS:
                move_draw_dest = move
            elif move.type == PLACE_ROUTE:
                claim_route_moves.append(move)
            elif move.type == DRAW_FACEUP or move.type == DRAW_FACEDOWN:
                draw_train_card_moves.append(move)
        
        total_current_points = 0
        for i in range(0, len(game.options.players)):
            total_current_points += game.options.players[i].points
        
        if self.players_previous_points < total_current_points:
            x = self.generate_game_plan(game)
            self.current_objective_route = [x[0], x[1]]
            self.current_objective_color = x[2]
            self.players_previous_points = total_current_points
        
        if self.current_objective_color != None:
            if self.current_objective_color == 'drawDestination':
                self.players_previous_points = -1
                return move_draw_dest

            for move in claim_route_moves:
                if (move.route.city1 == self.current_objective_route[0] and move.route.city2 == self.current_objective_route[1]) or (move.route.city1 == self.current_objective_route[1] and move.route.city2 == self.current_objective_route[0]):
                    return move
            
            draw_top_move = None
            draw_wild_move = None
            if self.current_objective_color != 'GRAY':
                for move in draw_train_card_moves:
                    if move.color == self.current_objective_color:
                        return move
                    if move.color == 'WILD':
                        draw_wild_move = move
                    if move.type == DRAW_FACEDOWN:
                        draw_top_move = move
            else:
                max_color = max(set(game.options.players[game.player_making_move].train_colors), key=game.options.players[game.player_making_move].train_colors.count)

                for move in draw_train_card_moves:
                    if move.color == max_color:
                        return move
                    if move.color == "WILD":
                        draw_wild_move = move
                    if move.type == DRAW_FACEDOWN:
                        draw_top_move = move
            
            if draw_wild_move != None:
                return draw_wild_move
            if draw_top_move != None:
                return draw_top_move
        
        if len(draw_train_card_moves) > 0:
            return Random.choice(Random(), draw_train_card_moves)
        if len(claim_route_moves) > 0:
            return Random.choice(Random(), claim_route_moves)
        
        return Random.choice(Random(), possible_moves)
    
    def generate_game_plan(self, game: GameEngine):
        joint_graph = self.joint_graph(game)
        city1 = None
        city2 = None
        color = None
        min_trains_threshold = 8

        list_of_destinations = self.destinations_not_completed(game, joint_graph)
        if list_of_destinations:
            most_valuable_route = None
            for destination in list_of_destinations:
                if most_valuable_route == None:
                    most_valuable_route = destination
                else:
                    if destination.points > most_valuable_route.points:
                        most_valuable_route = destination
            result = self.chooseNextRouteTarget(game, joint_graph, most_valuable_route)
            if result != False:
                city1, city2, color = result
        if city1 == None:
            min_number_of_trains = min([x.trains_left for x in game.options.players])
            if min_number_of_trains >= min_trains_threshold and len(game.destination_deck.cards) > 0:
                return ['drawDestination', 'drawDestination', 'drawDestination']
            else:
                result = self.chooseMaxRoute(game)
        return result
    
    def chooseMaxRoute(self, game: GameEngine):
        number_of_trains_left = game.options.players[game.player_making_move].trains_left
        max_size = 0
        list_of_edges = []

        free_routes_graph = self.free_routes_graph(game)
        for city1 in free_routes_graph:
            for city2 in free_routes_graph[city1]:
                for e in free_routes_graph[city1][city2]:
                    edge = free_routes_graph[city1][city2][e]
                    if edge['weight'] <= number_of_trains_left:
                        if edge['weight'] > max_size:
                            max_size = edge['weight']
                            list_of_edges = [(edge, city1, city2)]
                        elif edge['weight'] == max_size:
                            list_of_edges.append((edge, city1, city2))
        
        if len(list_of_edges) > 0:
            best_route = [self.rank(x[0], game) for x in list_of_edges]
            best_route = list_of_edges[best_route.index(max(best_route))]
            return [best_route[1], best_route[2], best_route[0]['color']]
        
        return [None, None, None]
    
    def rank(self, edge, game: GameEngine):
        color = edge['color']
        player_colors_no_wild = [color for color in game.options.players[game.player_making_move].train_colors if color != 'WILD']
        number_of_wilds = game.options.players[game.player_making_move].train_colors.count("WILD")
        color_list = set(player_colors_no_wild)
        max_color_value = max([player_colors_no_wild.count(x) for x in color_list])

        if color == 'GRAY':
            if max_color_value >= edge['weight']:
                return 10
            if max_color_value + number_of_wilds >= edge['weight']:
                return 9
            return 10 - edge['weight'] + max_color_value
        if player_colors_no_wild.count(color) >= edge['weight']:
            return 10
        if player_colors_no_wild.count(color) + number_of_wilds >= edge['weight']:
            return 9
        return 9 - edge['weight'] + max_color_value
    
    def chooseNextRouteTarget(self, game: GameEngine, graph: nx.MultiGraph, route: DestinationCard):
        try:
            list_of_route_nodes = nx.shortest_path(graph, route.city1, route.city2)
        except:
            return False

        list_of_colors = set()
        cities = []
        for i in range(0, len(list_of_route_nodes)-1):
            cities = [list_of_route_nodes[i], list_of_route_nodes[i+1]]
            for key in graph[list_of_route_nodes[i]][list_of_route_nodes[i+1]]:
                edge = graph[list_of_route_nodes[i]][list_of_route_nodes[i+1]][key]
                if edge['owner'] != None:
                    list_of_colors = set()
                    cities = []
                    break
                list_of_colors.add(edge['color'])
            if len(cities) != 0:
                break
        color_weight = []
        list_of_colors = list(list_of_colors)
        if 'GRAY' in list_of_colors:
            list_of_colors = list(set(game.options.players[game.player_making_move].train_colors))
        for color in list_of_colors:
            color_weight.append(game.options.players[game.player_making_move].train_colors.count(color))
        
        max_weight = color_weight.index(max(color_weight))
        desired_color = list_of_colors[max_weight]

        return [cities[0], cities[1], desired_color]

    def destinations_not_completed(self, game: GameEngine, joint_graph: nx.MultiGraph) -> list[DestinationCard]:
        result = []
        graph = nx.MultiGraph()
        graph.add_edges_from([edge for edge in game.board.edges(data=True) if edge[2]['owner'] == game.player_making_move])

        destination_cards = game.options.players[game.player_making_move].destinations
        for card in destination_cards:
            try:
                nx.shortest_path(graph, card.city1, card.city2)
                solved = True
            except:
                solved = False
            
            if not solved:
                if card.city1 in joint_graph and card.city2 in joint_graph and nx.has_path(joint_graph, card.city1, card.city2):
                    result.append(card)

        return result
    
    def chooseDestinations(self, moves: list[ChooseDestinations], game: GameEngine):
        best_move = (0, None)
        least_worst_move = (0, None)
        joint_graph = self.joint_graph(game)

        for m in moves:
            current_move_value = 0
            number_of_trains_needed = 0
            points = 0

            for destination in m.destinations:
                temp = self.calculate_value(destination, joint_graph)
                current_move_value += temp[0]
                number_of_trains_needed += temp[1]
                points += destination.points
            
            if number_of_trains_needed <= game.options.players[game.player_making_move].trains_left:
                total = current_move_value / number_of_trains_needed
                if total > best_move[0]:
                    best_move = (total, m)
            else:
                if least_worst_move[1] == None:
                    least_worst_move = (points, m)
                else:
                    if least_worst_move[0] > points:
                        least_worst_move = (points, m)
        
        if best_move[1] != None:
            return best_move[1]
        
        return least_worst_move[1]
    
    def calculate_value(self, destination: DestinationCard, graph: nx.MultiGraph):
        try:
            if nx.has_path(graph, destination.city1, destination.city2):
                left_to_claim = 0
                path = nx.shortest_path(graph, destination.city1, destination.city2)
                for s in range(0, len(path)-1):
                    for t in range(s+1, len(path)):
                        temp = 0
                        for edge in graph[path[s]][path[t]]:
                            if edge['owner'] == None:
                                temp = edge['weight']
                            else:
                                temp = 0
                                break
                    left_to_claim = left_to_claim + temp
                return [float(destination.points), float(left_to_claim)]
            else:
                return [float(-1.0*destination.points), float(50)]
        except:
            return [float(-1.0*destination.points), float(50)]
        
    
    def joint_graph(self, game: GameEngine):
        free_connections_graph = self.free_routes_graph(game)
        joint_graph = nx.MultiGraph()
        joint_graph.add_edges_from([edge for edge in game.board.edges(data=True) if edge[2]['owner'] == game.player_making_move])
        joint_graph.add_edges_from(free_connections_graph.edges(data=True))
        return joint_graph
    
    def free_routes_graph(self, game: GameEngine):
        graph = game.board
        number_of_players = len(game.options.players)

        G = nx.MultiGraph()
        visited_nodes = []

        for node1 in graph:
            for node2 in graph[node1]:
                if node2 not in visited_nodes:

                    # Filtering ineligible route pickups
                    locked = False
                    for edge in graph[node1][node2]:
                        if number_of_players < 4:
                            if graph[node1][node2][edge]['owner'] != None:
                                locked = True
                        else:
                            if graph[node1][node2][edge]['owner'] == game.player_making_move:
                                locked = True
                    
                    if not locked:
                        for edge in graph[node1][node2]:
                            if graph[node1][node2][edge]['owner'] == None:
                                G.add_edge(node1, node2, weight=graph[node1][node2][edge]['weight'], color=graph[node1][node2][edge]['color'], owner=None, index=graph[node1][node2][edge]['index'])
            
            visited_nodes.append(node1)

        return G