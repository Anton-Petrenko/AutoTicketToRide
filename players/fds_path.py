import collections
import networkx as nx
import queue
from engine.lib import *
from engine.game import GameEngine

class Path(Player):

    def __init__(self, name = "PathAgent"):
        super().__init__(name)
    
    def decide(self, game: GameEngine):
        possible_moves = game.get_valid_moves()
        if len(possible_moves) == 0:
            return None
        
        if possible_moves[0].type == CHOOSE_DESTINATIONS:
            for m in possible_moves:
                assert isinstance(m, ChooseDestinations)
                if len(m.destinations) == game.options.dests_dealt_per_player_start:
                    return m
        
        p_queue = queue.PriorityQueue()

        player_graph = nx.MultiGraph()
        player_graph.add_edges_from([edge for edge in game.board.edges(data=True) if edge[2]['owner'] == game.player_making_move])
        list_of_destinations = self.destinations_not_completed(game, player_graph)
        player_edges = player_graph.edges()
        joint_graph = self.free_routes_graph(game)
        for edge in player_edges:
            joint_graph.add_edge(edge[0], edge[1], weight=0, color='none')
        
        paths_to_take = []

        for destination in list_of_destinations:
            temp = []
            try:
                temp = nx.shortest_path(joint_graph, destination.city1, destination.city2)
            except:
                continue

            for i in range(0, len(temp)-1):
                if (temp[i], temp[i+1]) not in player_edges and (temp[i+1], temp[i]) not in player_edges:
                    if (temp[i], temp[i+1]) not in paths_to_take and (temp[i+1], temp[i]) not in paths_to_take:
                        weight = 0
                        try:
                            weight = game.board[temp[i]][temp[i+1]][0]['weight']
                        except:
                            weight = game.board.graph[temp[i]][temp[i+1]][0]['weight']

                        if weight < game.options.players[game.player_making_move].trains_left:
                            paths_to_take.append(((-1) * (POINTS_BY_LENGTH[weight]+2*destination.points), temp[i], temp[i+1]))

        free_connections_graph = self.free_routes_graph(game)

        if len(paths_to_take) == 0:
            for node1 in free_connections_graph:
                for node2 in free_connections_graph[node1]:
                    for key in free_connections_graph[node1][node2]:
                        if game.board[node1][node2][key]['weight'] < game.options.players[game.player_making_move].trains_left:
                            paths_to_take.append((POINTS_BY_LENGTH[game.board[node1][node2][key]['weight']], node1, node2))
        
        for path in paths_to_take:
            p_queue.put(path)
        
        first_max_color = None
        while not p_queue.empty():
            move = p_queue.get()
            color = []
            try:
                edges = free_connections_graph[move[1]][move[2]]
                for key in edges:
                    color.append(free_connections_graph[move[1]][move[2]][key]['color'])
            except:
                edges = free_connections_graph[move[2]][move[1]]
                for key in edges:
                    color.append(free_connections_graph[move[2]][move[1]][key]['color'])
            
            if 'GRAY' in color:
                color = ['RED', 'ORANGE', 'BLUE', 'PINK', 'WHITE', 'YELLOW', 'BLACK', 'GREEN']
            
            color_count = collections.Counter(game.options.players[game.player_making_move].train_colors)
            max_count = 0
            max_color = color[0]

            for c in color:
                if c in color_count:
                    if color_count[c] > max_count:
                        max_count = color_count[c]
                        max_color = c

            if first_max_color == None:
                first_max_color = max_color
            
            moves_available: list[PlaceRoute] = []
            for m in possible_moves:
                if m.type == PLACE_ROUTE:
                    assert isinstance(m, PlaceRoute)
                    if (m.route.city1 == move[1] and m.route.city2 == move[2]) or (m.route.city1 == move[2] and m.route.city2 == move[1]):
                        moves_available.append(m)
            
            if len(moves_available) > 0:
                return_move = None
                for m in moves_available:
                    if m.route.color == max_color:
                        return_move = m
                        break
                
                return return_move
            
        if len(game.traincolor_deck.cards) > 0:
            top_draw = None
            for m in possible_moves:
                if isinstance(m, DrawCard):
                    if m.color == first_max_color:
                        return m
                    elif m.type == DRAW_FACEDOWN:
                        top_draw = m
            return top_draw
        
        for m in possible_moves:
            if len(game.traincolor_deck.cards) > 0:
                if m.type == DRAW_FACEDOWN:
                    return m
            else:
                if isinstance(m, DrawCard) and m.color == 'WILD':
                    return m
        
        return Random.choice(Random(), possible_moves)
    
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

    def destinations_not_completed(self, game: GameEngine, graph: nx.MultiGraph) -> list[DestinationCard]:
        result = []
        for card in game.options.players[game.player_making_move].destinations:
            solved = False
            try:
                nx.shortest_path(graph, card.city1, card.city2)
                solved = True
            except:
                solved = False
            if not solved:
                result.append(card)
        return result