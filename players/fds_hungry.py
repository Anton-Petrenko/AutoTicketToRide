import networkx as nx
from engine.lib import Player
from engine.game import GameEngine

class Hungry(Player):

    def __init__(self, name = "HungryAgent"):
        super().__init__(name)

    def decide(self, game: GameEngine):
        possible_moves = game.get_valid_moves()
        free_connections_graph = self.free_routes_graph(game)
        joint_graph = nx.MultiGraph()
        joint_graph.add_edges_from([edge for edge in game.board.edges(data=True) if edge[2]['owner'] == game.player_making_move])
        joint_graph.add_edges_from(free_connections_graph.edges(data=True))
        
        d_points = 0 # earnable destination card points
        list_of_cities = [] # cities from completeable destination cards
        for destination in game.options.players[game.player_making_move].destinations:
            if destination.city1 in joint_graph and destination.city2 in joint_graph:
                list_of_cities.extend([destination.city1, destination.city2])
                d_points += destination.points
        



    def player_graph(self, game: GameEngine) -> nx.MultiGraph:
        G = nx.MultiGraph()
        G.add_edges_from([edge for edge in game.board.edges(data=True) if edge[2]['owner'] == game.player_making_move])
        return G

    def free_routes_graph(self, game: GameEngine) -> nx.MultiGraph:
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