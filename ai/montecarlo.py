import time
from engine import *
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
    def __init__(self, visits):
        self.visits = visits
        self.children: dict[Action, MonteCarloNode] = {}

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
        
        return SearchResults(wins, points, total_simulations)

    def find_action(self) -> Action:
        self.root = MonteCarloNode(0)
        valid_moves = self.game.get_valid_moves()

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