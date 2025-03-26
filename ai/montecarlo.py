import os
import time
from engine import *
from math import sqrt, log
import networkx as nx
import multiprocessing
from random import choice
from statistics import mean, median

WINS = 0
POINTS = 1
MEAN_SCORE = 2
MEDIAN_SCORE = 3
# NORMALIZED_SCORE = 4
# BOUNDED_NORMALIZED_SCORE = 5

class SearchResults:
    def __init__(self, wins: int, points: list[int], total_simulations: int):
        self.wins = wins
        self.points = points
        self.total_simulations = total_simulations

class MonteCarloNode:
    def __init__(self, visits, total_score = 0, heuristic_score = 0):
        self.visits = visits
        self.children: dict[Action, MonteCarloNode] = {}
        self.total_score = total_score
        self.heuristic_score = heuristic_score

class MonteCarloSearch:
    def __init__(self, game: GameEngine):
        self.game = game
    
    def search_space(self) -> Action:
        raise NotImplementedError()

class FlatMonteCarlo(MonteCarloSearch):
    def __init__(self, game: GameEngine, seconds_per_branch: int, simulation_goal: int):
        super().__init__(game)
        self.simulation_goal = simulation_goal
        self.seconds_per_branch = seconds_per_branch
    
    def search_branch(self, action: Action) -> SearchResults:
        # print(f"[{os.getpid()}] Beginning search")
        wins = 0
        points = []
        total_simulations = 0
        player_benefitting = self.game.player_making_move
        start_time = time.time()

        while time.time() - start_time < self.seconds_per_branch:
            game_clone = self.game.clone()
            game_clone.apply(action)
            current_node: MonteCarloNode = self.root.children[action]
            current_node.visits += 1

            while not game_clone.game_ended:
                valid_moves = game_clone.get_valid_moves()
                chosen_action = None if len(valid_moves) == 0 else choice(valid_moves)
                if chosen_action in current_node.children.keys():
                    current_node = current_node.children[chosen_action]
                    current_node.visits += 1
                else:
                    current_node.children[chosen_action] = MonteCarloNode(0)
                    current_node = current_node.children[chosen_action]
                    current_node.visits += 1
                
                game_clone.apply(chosen_action)
            
            points.append(game_clone.options.players[player_benefitting].points)
            if game_clone.final_standings[0].turn_order == player_benefitting: wins += 1
            total_simulations += 1
        
        # print(f"[{os.getpid()}] Search ended")
        return SearchResults(wins, points, total_simulations)

    def find_action(self) -> Action:
        self.root = MonteCarloNode(0)
        valid_moves = self.game.get_valid_moves()
        if len(valid_moves) == 0: return None

        for action in valid_moves:
            self.root.children[action] = MonteCarloNode(0)

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(self.search_branch, valid_moves)
        
        if self.simulation_goal == WINS:
            highest_index = None
            highest_wins = -1
            for i, results in enumerate(results):
                if results.wins > highest_wins:
                    highest_wins = results.wins
                    highest_index = i
            return valid_moves[highest_index]
        elif self.simulation_goal == POINTS:
            highest_index = None
            highest_points = None
            for i, results in enumerate(results):
                sum_points = sum(results.points)
                if highest_points == None or sum_points > highest_points:
                    highest_points = sum_points
                    highest_index = i
            return valid_moves[highest_index]
        elif self.simulation_goal == MEAN_SCORE:
            highest_index = None
            highest_mean = None
            for i, results in enumerate(results):
                mean_points = mean(results.points)
                if highest_mean == None or mean_points > highest_mean:
                    highest_mean = mean_points
                    highest_index = i
            return valid_moves[highest_index]
        elif self.simulation_goal == MEDIAN_SCORE:
            highest_index = None
            highest_median = None
            for i, results in enumerate(results):
                median_points = median(results.points)
                if highest_median == None or median_points > highest_median:
                    highest_median = median_points
                    highest_index = i
            return valid_moves[highest_index]

class MonteCarloPUB(MonteCarloSearch):
    def __init__(self, game: GameEngine, seconds_per_branch: int, uct_c_constant: float = 1.41421356237):
        super().__init__(game)
        self.seconds_per_branch = seconds_per_branch
        self.uct_c_constant = uct_c_constant
    
    def get_needed_edge_ids(self):
        # print(f"SEARCH FOR TURN {self.game.turn}")
        player_graph = nx.MultiGraph()
        player_graph.add_edges_from([edge for edge in self.game.board.edges(data=True) if edge[2]['owner'] == self.game.player_making_move])

        available_routes_graph = nx.MultiGraph()
        available_routes_graph.add_edges_from([edge for edge in self.game.board.edges(data=True) if edge[2]['owner'] == None])

        joint_graph = nx.MultiGraph()
        joint_graph.add_edges_from([edge for edge in self.game.board.edges(data=True) if edge[2]['owner'] == self.game.player_making_move])
        joint_graph.add_edges_from([edge for edge in self.game.board.edges(data=True) if edge[2]['owner'] == None])

        edge_ids_needed = []

        for destination_card in self.game.options.players[self.game.player_making_move].destinations:
            if joint_graph.has_node(destination_card.city1) and joint_graph.has_node(destination_card.city2):
                if nx.has_path(joint_graph, destination_card.city1, destination_card.city2):
                    # print(destination_card)
                    # The destination card has been determined to be claimable, start by getting the shortest available routes
                    cur_shortest_paths: list[list[str]] = list(nx.all_shortest_paths(joint_graph, destination_card.city1, destination_card.city2, weight="weight"))
                    cur_shortest_length = nx.shortest_path_length(joint_graph, destination_card.city1, destination_card.city2)
                    final_shortest_paths: list[list[str]] = []
                    path_added = False
                    for city in player_graph:
                        if city == destination_card.city1 or city == destination_card.city2: continue
                        # If the player has cities on any of the shortest paths, we prioritize those paths only
                        for path in cur_shortest_paths:
                            if city in path:
                                if not path_added: 
                                    final_shortest_paths.append(path)
                                    path_added = True
                        # The player may have no cities on any of the shortest paths
                        if len(final_shortest_paths) == 0:
                            if nx.has_path(joint_graph, city, destination_card.city1) and nx.has_path(joint_graph, city, destination_card.city2):
                                if nx.shortest_path_length(joint_graph, city, destination_card.city1, weight="weight") + nx.shortest_path_length(joint_graph, city, destination_card.city2, weight="weight") <= cur_shortest_length:
                                    for path1 in nx.all_shortest_paths(joint_graph, city, destination_card.city1, weight="weight"):
                                        for path2 in nx.all_shortest_paths(joint_graph, city, destination_card.city2, weight="weight"):
                                            final_shortest_paths.extend(path1 + path2)
                    if len(final_shortest_paths) == 0:
                        final_shortest_paths.extend(cur_shortest_paths)
                    # print(final_shortest_paths)
                    for path in final_shortest_paths:
                        for i in range(len(path)-1):
                            for edge in available_routes_graph.edges(data=True):
                                if (edge[0] == path[i] and edge[1] == path[i+1]) or (edge[0] == path[i+1] and edge[1] == path[i]):
                                    # print(edge)
                                    edge_ids_needed.append(edge[2]["index"])

        return edge_ids_needed

    def get_needed_colors(self, edges: list[int]):
        ret = []
        edges = list(set(edges))
        for edge in self.game.board.edges(data=True):
            if edge[2]["index"] in edges:
                if edge[2]["color"] != "GRAY" and edge[2]["color"] not in ret:
                    ret.append(edge[2]["color"])
        return ret

    def heuristic_values(self, actions: list[Action]) -> list[tuple[Action, float]]:
        edge_ids_needed = self.get_needed_edge_ids()
        colors_needed = self.get_needed_colors(edge_ids_needed)
        ret = []
        for action in actions:
            if action.type == PLACE_ROUTE:
                assert isinstance(action, PlaceRoute)
                count = edge_ids_needed.count(action.route.id)
                value = -10 if count == 0 else count*10
                ret.append((action, value))
            elif action.type == DRAW_FACEUP:
                assert isinstance(action, DrawCard)
                assert action.color != None
                value = 0
                if action.color == "WILD": value = 1
                if action.color in colors_needed: value = 4
                ret.append((action, value))
            elif action.type == DRAW_FACEDOWN:
                ret.append((action, -4))
            else:
                ret.append((action, 0))
        return ret

    def find_action(self) -> Action:
        self.root = MonteCarloNode(0)
        valid_moves = self.game.get_valid_moves()
        if len(valid_moves) == 0: return None

        action_to_value: list[tuple[Action, float]] = self.heuristic_values(valid_moves)
        assert len(action_to_value) == len(valid_moves)

        for action, value in action_to_value:
            self.root.children[action] = MonteCarloNode(0, heuristic_score=value)

        start_time = time.time()
        cur_node = self.root
        while time.time() - start_time < self.seconds_per_branch:

            # Select until a leaf node
            path: list[MonteCarloNode] = []
            while len(cur_node.children) > 0:
                # Calculate UCT score for each child
                max_uct_score = None
                max_uct_node = None
                for action, child in cur_node.children.items():
                    uct_score = (child.total_score / (child.visits+1)) + (self.uct_c_constant*sqrt(log(cur_node.visits+1)/(child.visits+1))) + ((child.heuristic_score)/(child.visits+1))
                    if max_uct_score == None:
                        max_uct_score = uct_score
                        max_uct_node = child
                    if max_uct_score < uct_score:
                        max_uct_score = uct_score
                        max_uct_node = child
                assert isinstance(max_uct_node, MonteCarloNode)
                max_uct_node.visits += 1
                path.append(max_uct_node)
                cur_node = max_uct_node
            
            # Playout randomly
            game_clone = self.game.clone()
            
            while game_clone.game_ended:
                valid_clone_moves = game_clone.get_valid_moves()
                chosen_action = None if len(valid_clone_moves) == 0 else choice(valid_clone_moves)
                if chosen_action in cur_node.children.keys():
                    cur_node = cur_node.children[chosen_action]
                    cur_node.visits += 1
                else:
                    cur_node.children[chosen_action] = MonteCarloNode(0)
                    cur_node = cur_node.children[chosen_action]
                    cur_node.visits += 1
                path.append(cur_node)
                game_clone.apply(chosen_action)

            # Backprop
            for node in path:
                node.total_score += game_clone.options.players[self.game.player_making_move].points
            
            cur_node = self.root
        
        max_visits = -1
        max_action = None
        for action, child in self.root.children.items():
            if child.visits > max_visits:
                max_action = action
                max_visits = child.visits
        
        return max_action