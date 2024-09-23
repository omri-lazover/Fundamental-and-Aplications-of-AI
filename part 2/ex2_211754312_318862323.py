import logging
import json
#import networkx as nx
import random
import math
import itertools
import copy
from copy import deepcopy
from typing import List
from collections import deque
import numpy as np
import time
from itertools import product

import utils

ids = ["211754312", "318862323"]

RESET_PENALTY = 2
DROP_IN_DESTINATION_REWARD = 4
INIT_TIME_LIMIT = 300
TURN_TIME_LIMIT = 0.1
MARINE_COLLISION_PENALTY = 1



#-----------------OPTIMAL-----------------------
class OptimalPirateAgent:

    def treasures_combinations(self, state, total_opts):
        treasures_dict = state['treasures']
        for t_name, t_info in treasures_dict.items():
            possible_locs = []
            for possible_location in t_info['possible_locations']:
                prob_change_location = t_info['prob_change_location'] / len(t_info['possible_locations'])
                if possible_location == t_info['location']:
                    possible = prob_change_location + (1 - t_info['prob_change_location'])
                else:
                    possible = prob_change_location
                possible_locs.append((t_name, possible_location, possible))
            total_opts.append(possible_locs)
        return total_opts

    def marine_ships_combinations(self, state, total_opts):
        marine_ships = state['marine_ships']
        for marine_name, marine in marine_ships.items():
            marine_options = []
            potential_indices = [marine['index']]
            if marine['index'] > 0:
                potential_indices.append(marine['index'] - 1)
            if marine['index'] < len(marine['path']) - 1:
                potential_indices.append(marine['index'] + 1)
            for possible_index in potential_indices:
                marine_options.append((marine_name, possible_index, 1 / len(potential_indices)))
            total_opts.append(marine_options)
        return total_opts

    def final_combinations(self, total_opts):
        all_combinations = list(itertools.product(*total_opts))

        all_combos_with_probs = []
        for c in all_combinations:
            probabilities = [item[-1] for item in c]
            prod = 1
            for p in probabilities:
                prod *= p
            prod = round(prod, 6)
            all_combos_with_probs.append((prod,) + c)

        return all_combos_with_probs

    def generate_all_combinations_for_state(self, state):
        total_opts = []

        total_opts = self.treasures_combinations(state, total_opts)

        total_opts = self.marine_ships_combinations(state, total_opts)

        return self.final_combinations(total_opts)

    def marines_locations(self, state):
        marine_positions = []
        for marine in state['marine_ships'].values():
            marine_positions.append(marine['path'][marine['index']])
        return marine_positions

    def reset_board(self, state):
        copied_init = copy.deepcopy(self.initial)
        action_score_diff = -2

        marine_positions = []
        for marine in state['marine_ships'].values():
            marine_positions.append(marine['path'][marine['index']])

        for ship_name, ship in copied_init['pirate_ships'].items():

            if ship['location'] in marine_positions:
                action_score_diff -= 1
                copied_init['pirate_ships'][ship_name]['capacity'] = self.initial['pirate_ships'][ship_name]['capacity']

        return [(self.shrinked_state(copied_init), 1, action_score_diff)]

    def probs_for_atomic_action(self, state, action):
        #state_after_action = copy.deepcopy(state)
        for atomic_action in action:
            if atomic_action[0] == 'sail':
                state['pirate_ships'][atomic_action[1]]['location'] = atomic_action[2]

            if atomic_action[0] == 'collect':
                state['pirate_ships'][atomic_action[1]]['capacity'] -= 1

            if atomic_action[0] == 'deposit':
                state['pirate_ships'][atomic_action[1]]['capacity'] = \
                self.initial['pirate_ships'][atomic_action[1]]['capacity']
        return self.generate_all_combinations_for_state(state)

    def state_prob_score(self, state, action):
        if action == "reset":
            return self.reset_board(state)


        state_after_action = copy.deepcopy(state)
        atomic_action_probs_combinations = self.probs_for_atomic_action(state_after_action, action)

        state_changes_with_probs = []
        for comb in atomic_action_probs_combinations:
            new_state = copy.deepcopy(state_after_action)
            for state_info in comb[1:]:
                if state_info[0] in self.treasures_names:
                    new_state['treasures'][state_info[0]]['location'] = state_info[1]
                else:
                    if state_info[0] in self.marines_names:
                        new_state['marine_ships'][state_info[0]]['index'] = state_info[1]

            marines_locations = self.marines_locations(new_state)

            state_points = 0

            for ship_name, ship in new_state['pirate_ships'].items():

                if ship['location'] in marines_locations:
                    state_points -= 1
                    new_state['pirate_ships'][ship_name]['capacity'] = self.initial['pirate_ships'][ship_name]['capacity']

            for atomic_action in action:
                if atomic_action[0] == "deposit":
                    state_points += 4 * (self.initial['pirate_ships'][atomic_action[1]]['capacity'] - state['pirate_ships'][atomic_action[1]]['capacity'])

            state_changes_with_probs += [(self.shrinked_state(new_state), comb[0], state_points)]
        return state_changes_with_probs

    def policy_find(self):
        value_iteration_matrix = {i: {json.dumps(state): 0 for state in self.states} for i in range(self.total_turns + 1)}
        state_actions = {json.dumps(state): self.actions_for_state(state) for state in self.states}

        state_action_p_dict = \
            {json.dumps(state): {action: [] for action in self.actions_for_state(state)}
             for state in self.states}

        for state in self.states:
            for action in self.actions_for_state(state):
                state_action_p_dict[json.dumps(state)][action] = self.state_prob_score(state, action)

        optimal_strategy = {json.dumps(state): {turn: None for turn in range(1, self.total_turns + 1)} for
                        state in self.states}
        for turn_index in range(1, self.total_turns + 1):
            for state in self.states:
                max_value = float('-inf') # value of the state
                optimal_action = None # action that maximizes the value
                actions_for_state = state_actions[json.dumps(state)]
                for action in actions_for_state:
                    action_val = 0 # value of the action
                    for next_state in state_action_p_dict[json.dumps(state)][action]:
                        s_prime, p, score_from_action = next_state[0], next_state[1], next_state[2]
                        action_val += p * ((value_iteration_matrix[turn_index - 1][json.dumps(s_prime)]) + score_from_action)
                    if action_val > max_value:
                        max_value = action_val
                        optimal_action = action
                value_iteration_matrix[turn_index][json.dumps(state)] = max_value


                optimal_strategy[json.dumps(state)][turn_index] = optimal_action

        return optimal_strategy

    def __init__(self, initial):
        self.initial = initial
        self.total_turns = initial["turns to go"]
        self.treasures_names = list(initial["treasures"].keys())
        self.marines_names = list(initial["marine_ships"].keys())
        self.base_location = next(iter(initial['pirate_ships'].values()), None).get("location")
        self.map = initial['map']
        self.curr_turn = 0

        self.graph = self.build_graph()
        self.states = self.all_states(self.initial)
        self.policy = self.policy_find()


    def act(self, state):
        self.curr_turn = self.curr_turn + 1
        return self.policy[json.dumps(self.shrinked_state(state))][self.total_turns-self.curr_turn+1]


    def build_graph(self):
        """
        build the graph of the problem
        """
        n, m = len(self.map), len(
            self.map[0])
        g = nx.grid_graph((m, n))
        nodes_to_remove = []
        for node in g:
            if self.map[node[0]][node[1]] == 'I':
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            g.remove_node(node)
        return g


    def all_states(self, state):
        pirate_ships = state["pirate_ships"]
        treasures = state["treasures"]
        marine_ships = state["marine_ships"]

        pirate_locations = list(self.graph.nodes())


        pirates_possible_locations, capacities = list(itertools.product(pirate_locations, repeat=len(pirate_ships))), [range(pirate_ship["capacity"] + 1) for pirate_ship in pirate_ships.values()]

        all_marines_possible_combs = itertools.product(*[range(len(marine_ship["path"])) for marine_ship in marine_ships.values()])

        all_treasures_possible_combs = itertools.product(
            *[treasure["possible_locations"] for treasure in treasures.values()])

        all_states = []

        for pirate_combo, capacity_combo, treasure_combo, marine_index_combo in itertools.product(
                pirates_possible_locations,
                itertools.product(*capacities),
                all_treasures_possible_combs,
                all_marines_possible_combs):
            new_state = copy.deepcopy(state)

            for (location, capacity), pirate_ship_name in zip(zip(pirate_combo, capacity_combo), pirate_ships.keys()):
                new_state["pirate_ships"][pirate_ship_name]["location"] = location
                new_state["pirate_ships"][pirate_ship_name]["capacity"] = capacity

            for location, treasure_name in zip(treasure_combo, treasures.keys()):
                new_state["treasures"][treasure_name]["location"] = location

            for index, marine_ship_name in zip(marine_index_combo, marine_ships.keys()):
                new_state["marine_ships"][marine_ship_name]["index"] = index

            all_states.append(self.shrinked_state(new_state))
        return all_states

    def find_treasure_name_by_location(self, location, state):
        treasures = []
        for treasure_name, treasure_location in state['treasures'].items():
            if treasure_location['location'] == location:
                treasures.append(treasure_name)
        return treasures

    def legal_neighbors(self, x, y):
        adj = []
        if x > 0 and self.map[x - 1][y] != 'I':
            adj.append([x - 1, y])
        if x < len(self.map) - 1 and self.map[x + 1][y] != 'I':
            adj.append([x + 1, y])
        if y > 0 and self.map[x][y - 1] != 'I':
            adj.append([x, y - 1])
        if y < len(self.map) - 1 and self.map[x][y + 1] != 'I':
            adj.append([x, y + 1])
        return adj

    def actions_for_state(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        treasures_location = []
        for treasures in state['treasures'].values():
            treasures_location.append(treasures['location'])

        all_ship_actions = []
        for ship_name, ship_dict in state['pirate_ships'].items():
            pirate_ship_actions = self.actions_per_ship(ship_name, ship_dict['location'], state, treasures_location)
            all_ship_actions.append(pirate_ship_actions)

        all_possible_actions_combinations = list(itertools.product(*all_ship_actions))
        all_possible_actions_combinations += ["reset"]
        all_possible_actions_combinations += ["terminate"]
        return all_possible_actions_combinations

    def actions_per_ship(self, ship, location, state, treasures_location):
        actions = []
        possible_neighbors = self.legal_neighbors(location[0], location[1])

        # deposit
        if state['pirate_ships'][ship]['location'] == self.base_location \
                and state['pirate_ships'][ship]['capacity'] < self.initial['pirate_ships'][ship]['capacity']:
            actions += [("deposit", ship)]

        # collect
        if state['pirate_ships'][ship]['capacity'] > 0:
            treasure_names = self.find_treasure_name_by_location((location[0] + 1, location[1]), state)
            for treasure_name in treasure_names:
                if location[0] + 1 < len(self.map) and any(
                        (location[0] + 1, location[1]) == coords for coords in treasures_location):
                    actions += [("collect", ship, treasure_name)]
            treasure_names = self.find_treasure_name_by_location((location[0] - 1, location[1]), state)
            for treasure_name in treasure_names:
                if location[0] - 1 >= 0 and any(
                        (location[0] - 1, location[1]) == coords for coords in treasures_location):
                    actions += [("collect", ship, treasure_name)]

            treasure_names = self.find_treasure_name_by_location((location[0], location[1] + 1), state)
            for treasure_name in treasure_names:
                if location[1] + 1 < len(self.map[0]) and any(
                        (location[0], location[1] + 1) == coords for coords in treasures_location):
                    actions += [("collect", ship, treasure_name)]

            treasure_names = self.find_treasure_name_by_location((location[0], location[1] - 1), state)
            for treasure_name in treasure_names:
                if location[1] - 1 >= 0 and any(
                        (location[0], location[1] - 1) == coords for coords in treasures_location):
                    actions += [("collect", ship, treasure_name)]

        # sail
        for neighbor in possible_neighbors:
            actions += [("sail", ship, (neighbor[0], neighbor[1]))]

        actions += [("wait", ship)]
        return actions

    def shrinked_state(self, original_dict):
        keys = ["pirate_ships", "treasures", "marine_ships"]
        new_dict = {key: original_dict[key] for key in keys if key in original_dict}
        return new_dict





#-----------------NOT OPTIMAL----------------------------

class PirateAgent:

    def simulation(self, probPolicy=0, policy=None):

        current_state = self.shrinked_state(self.initial)
        simulation_dict = {}

        state_action_revenues = [0] *self.total_turns

        for round in range(self.total_turns, 0, -1):
            selected_action = None

            if policy is not None:
                if random.random() < probPolicy:
                    state_key = json.dumps(current_state, sort_keys=True)
                    if state_key in policy:
                        selected_action = policy[state_key]["best_action"]

            if selected_action is None:
                actions_list = self.actions_for_state(current_state)

                if random.random() > 0.5:
                    capacity = current_state["pirate_ships"][self.pirate_names[0]]["capacity"]
                    if capacity == 2:
                        selected_action = self.h(current_state, actions_list, False, current_state["treasures"])
                    elif capacity == 1:
                        selected_action = self.h(current_state, actions_list, True, current_state["treasures"])
                    else:
                        selected_action = self.h(current_state, actions_list, True, False)

                if selected_action is None:
                    for action in actions_list:
                        if action[0][0] == "deposit" and selected_action is None:
                            selected_action = action
                        elif action[0][0] == "collect":
                            if random.random() > 0.05:
                                selected_action = action


                if selected_action is None:
                    selected_action = random.choice(actions_list)

            result = self.play_action(current_state, selected_action)
            next_state, points = result[0]
            state_action_revenues[round-1] = [current_state, selected_action, points]

            current_state = next_state

        for turn_num in range(1, self.total_turns-1):
            state, action, points = state_action_revenues[turn_num-1]
            state_key = json.dumps(state, sort_keys=True)
            state_action_revenues[turn_num][2] += points

            if state_key not in simulation_dict:
                simulation_dict[state_key] = {}
            if action not in simulation_dict[state_key]:
                simulation_dict[state_key][action] = {'points': points/turn_num, 'count': 1}
            else:
                simulation_dict[state_key][action]['points'] += points/turn_num
                simulation_dict[state_key][action]['count'] += 1

        turn_num=self.total_turns
        state, action, points = state_action_revenues[turn_num-1]
        state_key = json.dumps(state, sort_keys=True)

        if state_key not in simulation_dict:
            simulation_dict[state_key] = {}
        if action not in simulation_dict[state_key]:
            simulation_dict[state_key][action] = {'points': points / turn_num, 'count': 1}
        else:
            simulation_dict[state_key][action]['points'] += points / turn_num
            simulation_dict[state_key][action]['count'] += 1

        return simulation_dict

    def run_simulations(self, runs_for_single_sim=1000, total_simulations=15):
        processed_data_dict = {}  # Dictionary to store the processed data
        aggregated_results_dict = {}  # Dictionary to aggregate results from all simulations
        lower_bound, upper_bound = 0.2, 0.8
        lower_bound, upper_bound = 0.2, 0.95
        use_policy_prob=0

        policy = None
        for sim in range(total_simulations):
            if sim>0:
                use_policy_prob = lower_bound+(upper_bound-lower_bound)*(sim-1)/(total_simulations-2)
                policy = processed_data_dict
            for i in range(runs_for_single_sim):
                curr_sim_result = self.simulation(use_policy_prob, policy)

                for state_key, actions in curr_sim_result.items():
                    if state_key not in aggregated_results_dict:
                        aggregated_results_dict[state_key] = {}

                    for action, metrics in actions.items():
                        if action not in aggregated_results_dict[state_key]:
                            aggregated_results_dict[state_key][action] = {'points': 0, 'count': 0}

                        aggregated_results_dict[state_key][action]['points'] += metrics['points']
                        aggregated_results_dict[state_key][action]['count'] += metrics['count']

            for state_key, actions in aggregated_results_dict.items():
                processed_data_dict[state_key] = {}


                max_ratio = float('-inf')
                best_action = None

                for action, metrics in actions.items():
                    points = metrics['points']
                    count = metrics['count']

                    if count > 0:
                        ratio = points / count
                        if ratio > max_ratio:
                            max_ratio = ratio
                            best_action = action


                processed_data_dict[state_key]['best_action'] = best_action
                processed_data_dict[state_key]['max_points_count_ratio'] = max_ratio

            t2 = time.time()
            if t2 - self.t > 275:
                break

        return processed_data_dict


    def __init__(self, initial):
        self.t = time.time()
        self.initial = initial
        self.total_turns = initial["turns to go"]
        self.treasures_names = list(initial["treasures"].keys())
        self.marines_names = list(initial["marine_ships"].keys())
        self.map = initial['map']

        self.pirates_num=len(initial["pirate_ships"])
        self.pirate_names=[key for key in initial["pirate_ships"].keys()]
        self.pirate_for_example=self.pirate_names[0]
        self.base_location = initial["pirate_ships"][self.pirate_for_example]["location"]



        self.bfs_maps= self.bfs_all_combinations(self.initial["treasures"])




        self.policy = self.run_simulations()

    def act(self, state):
        action="terminate"
        shrinked_state=self.shrinked_state(state)
        treasures = shrinked_state["treasures"]
        state_to_key=json.dumps(shrinked_state, sort_keys=True)


        if state_to_key in self.policy:
            action=self.policy[state_to_key]["best_action"]
            return self.duplicate_action_for_all_ships(action)
        else:
            possible_actions = self.actions_for_state(shrinked_state)
            capacity = shrinked_state["pirate_ships"][self.pirate_for_example]["capacity"]

            if capacity>0:
                action=self.h(shrinked_state, possible_actions, (capacity<2), treasures)
            else:
                action = self.h(shrinked_state, possible_actions, True, False)

        action=self.duplicate_action_for_all_ships(action)
        return action

    def find_treasure_name_by_location(self, location, state):
        treasures = []
        for treasure_name, treasure_location in state['treasures'].items():
            if treasure_location['location'] == location:
                treasures.append(treasure_name)
        return treasures

    def legal_neighbors(self, x, y):
        # neighbors that are not islands
        adj = []
        if x > 0 and self.map[x - 1][y] != 'I':
            adj.append([x - 1, y])
        if x < len(self.map) - 1 and self.map[x + 1][y] != 'I':
            adj.append([x + 1, y])
        if y > 0 and self.map[x][y - 1] != 'I':
            adj.append([x, y - 1])
        if y < len(self.map) - 1 and self.map[x][y + 1] != 'I':
            adj.append([x, y + 1])
        return adj

    def actions_for_state(self, state):
        treasures_location = []
        for treasures in state['treasures'].values():
            treasures_location.append(treasures['location'])

        all_ship_actions = []
        for ship_name, ship_dict in state['pirate_ships'].items():
            pirate_ship_actions = self.actions_per_ship(ship_name, ship_dict['location'], state, treasures_location)
            all_ship_actions.append(pirate_ship_actions)

        all_possible_actions_combinations = list(itertools.product(*all_ship_actions))
        all_possible_actions_combinations += ["reset"]
        return all_possible_actions_combinations

    def actions_per_ship(self, ship, location, state, treasures_location):
        actions = []
        possible_neighbors = self.legal_neighbors(location[0], location[1])

        # deposit
        if state['pirate_ships'][ship]['location'] == self.base_location \
                and state['pirate_ships'][ship]['capacity'] < self.initial['pirate_ships'][ship]['capacity']:
            actions += [("deposit", ship)]

        # collect
        if state['pirate_ships'][ship]['capacity'] > 0:
            treasure_names = self.find_treasure_name_by_location((location[0] + 1, location[1]), state)
            for treasure_name in treasure_names:
                if location[0] + 1 < len(self.map) and any(
                        (location[0] + 1, location[1]) == coords for coords in treasures_location):
                    actions += [("collect", ship, treasure_name)]
            treasure_names = self.find_treasure_name_by_location((location[0] - 1, location[1]), state)
            for treasure_name in treasure_names:
                if location[0] - 1 >= 0 and any(
                        (location[0] - 1, location[1]) == coords for coords in treasures_location):
                    actions += [("collect", ship, treasure_name)]

            treasure_names = self.find_treasure_name_by_location((location[0], location[1] + 1), state)
            for treasure_name in treasure_names:
                if location[1] + 1 < len(self.map[0]) and any(
                        (location[0], location[1] + 1) == coords for coords in treasures_location):
                    actions += [("collect", ship, treasure_name)]

            treasure_names = self.find_treasure_name_by_location((location[0], location[1] - 1), state)
            for treasure_name in treasure_names:
                if location[1] - 1 >= 0 and any(
                        (location[0], location[1] - 1) == coords for coords in treasures_location):
                    actions += [("collect", ship, treasure_name)]

        # sail
        for neighbor in possible_neighbors:
            actions += [("sail", ship, (neighbor[0], neighbor[1]))]

        actions += [("wait", ship)]
        return actions

    def environment_step(self, state):
        """
        update the state of environment randomly
        """
        for t in state['treasures']:
            treasure_stats = state['treasures'][t]
            if random.random() < treasure_stats['prob_change_location']:
                treasure_stats['location'] = random.choice(
                    treasure_stats['possible_locations'])

        for marine in state['marine_ships']:
            marine_stats = state["marine_ships"][marine]
            index = marine_stats["index"]
            if len(marine_stats["path"]) == 1:
                continue
            if index == 0:
                marine_stats["index"] = random.choice([0, 1])
            elif index == len(marine_stats["path"]) - 1:
                marine_stats["index"] = random.choice([index, index - 1])
            else:
                marine_stats["index"] = random.choice(
                    [index - 1, index, index + 1])
        return

    def duplicate_action_for_all_ships(self, action):
        if (action in ["reset", "terminate"]):
            return action
        action=action[0]
        if (action[0] in ["sail", "collect"]):
            return tuple([(action[0], self.pirate_names[i], action[2]) for i in range(self.pirates_num)])
        return tuple([(action[0], self.pirate_names[i]) for i in range(self.pirates_num)])


    def generate_treasure_matrix(self, num_treasures, prob):
        prob_move = prob / num_treasures
        p_not_move = (1 - prob) + prob_move
        matrix = [[prob_move if i != j else p_not_move for j in range(num_treasures)] for i in range(num_treasures)]
        return matrix

    def sssp_from_base(self, source, n_prob=None):
        queue = utils.FIFOQueue()
        visited = set()
        rule = False
        if n_prob is not None:
            rule = True
            t_matrix = self.generate_treasure_matrix(n_prob[0], n_prob[1])
            t_change_matrix = []
            for row in t_matrix:
                t_change_matrix.append(row.copy())
            t_count = 1

        new_map = [[-1 for i in range(len(row))] for row in self.map]

        new_map[source[0]][source[1]] = 1
        distance = 1
        queue.append((source, distance))
        visited.add(source)

        while len(queue) > 0:
            location, distance = queue.pop()
            travel = []
            if (location[0] > 0):
                travel.append((location[0] - 1, location[1]))
            if (location[1] > 0):
                travel.append((location[0], location[1] - 1))
            if (location[0] < len(self.map) - 1):
                travel.append((location[0] + 1, location[1]))
            if (location[1] < len(self.map[0]) - 1):
                travel.append((location[0], location[1] + 1))

            for new_location in travel:
                if new_location not in visited:
                    visited.add(new_location)
                    symbol = self.map[new_location[0]][new_location[1]]
                    if symbol == "S" or symbol == "B":
                        queue.append((new_location, distance + 1))
                        if rule:
                            while distance > t_count:
                                t_change_matrix = utils.matrix_multiplication(t_change_matrix, t_matrix)
                                t_count += 1
                            new_map[new_location[0]][new_location[1]] = t_change_matrix[0][0]
                        else:
                            new_map[new_location[0]][new_location[1]] = (1 / distance)
        if rule == False:
            self.base_diameter = 2 * distance
        return new_map

    def max_map(self, map_list):
        n, m = len(map_list[0]), len(map_list[0][0])
        max_values_map = [[0 for j in range(m)] for i in range(n)]
        for i in range(n):
            for j in range(m):
                max_values_map[i][j] = max(map_list, key=lambda x: x[i][j])[i][j]
        return max_values_map

    def shrinked_state(self, original_dict):
        stochastic_keys = ["treasures", "marine_ships"]
        shrinked_dict = {key: original_dict[key] for key in stochastic_keys if key in original_dict}
        shrinked_dict["pirate_ships"] = {self.pirate_for_example: original_dict["pirate_ships"][self.pirate_for_example]}
        return shrinked_dict

    def reset_board(self):
        new_state = copy.deepcopy(self.initial)
        point_to_change = -2

        marine_ship_locations_list = []
        for marine in new_state['marine_ships'].values():
            marine_ship_locations_list.append(marine['path'][marine['index']])

        for ship_name, ship in new_state['pirate_ships'].items():

            if ship['location'] in marine_ship_locations_list:
                point_to_change -= 1
                new_state['pirate_ships'][ship_name]['capacity'] = self.initial['pirate_ships'][ship_name][
                    'capacity']

        return [(self.shrinked_state(new_state), point_to_change)]

    def play_action(self, current_state, action):
        state_after_turn = copy.deepcopy(current_state)
        self.environment_step(state_after_turn)


        if action == "reset":
            return self.reset_board()

        else:
            score_diff = 0
            marines_next_locations = []
            for marine in state_after_turn['marine_ships'].values():
                marines_next_locations.append(marine['path'][marine['index']])

            for act in action:
                if act[0] == 'sail':
                    state_after_turn['pirate_ships'][act[1]]['location'] = act[2]

                elif act[0] == 'collect':
                    state_after_turn['pirate_ships'][act[1]]['capacity'] -= 1

                elif act[0] == 'deposit':
                    score_diff += self.pirates_num * 4 * \
                                  (self.initial['pirate_ships'][act[1]]['capacity'] -
                                   current_state['pirate_ships'][act[1]]['capacity'])
                    state_after_turn['pirate_ships'][act[1]]['capacity'] = \
                        self.initial['pirate_ships'][act[1]]['capacity']

            for p_name, p_info in state_after_turn['pirate_ships'].items():
                if p_info['location'] in marines_next_locations:
                    score_diff -= self.pirates_num
                    state_after_turn['pirate_ships'][p_name]['capacity'] = self.initial['pirate_ships'][p_name][
                        'capacity']

            return [(self.shrinked_state(state_after_turn), score_diff)]


    def bfs_all_combinations(self, treasure_dict):
        lengths = []
        names = []
        total_combs = 1
        for treasure in treasure_dict:
            names.append(treasure)
            length = len(treasure_dict[treasure]["possible_locations"])
            lengths.append(length)
            total_combs *= length

        base_sssp = self.sssp_from_base(self.base_location)
        treasure_sssp_dict = {"base": base_sssp}
        total_num_treasures = len(names)



        for i in range(total_combs):
            created_bfs_list = []
            names_locations = utils.hashabledict()
            names_locations["base"] = False
            curr_iter = i
            for j in range(total_num_treasures):
                name = names[j]
                l = lengths[j]
                prob = treasure_dict[name]["prob_change_location"]
                idx = curr_iter % l
                treasure_location = treasure_dict[name]["possible_locations"][idx]
                names_locations[name] = treasure_location
                created_bfs_list.append(self.sssp_from_base(treasure_location, (l, prob)))
                curr_iter //= l

            treasure_map = self.max_map(created_bfs_list)
            treasure_sssp_dict[names_locations] = treasure_map
            names_locations = copy.deepcopy(names_locations)
            names_locations["base"] = True
            treasure_map = [treasure_map, base_sssp]
            treasure_sssp_dict[names_locations] = self.max_map(treasure_map)

        return treasure_sssp_dict

    def base_treasures_dict(self, treasure_dict, base):
        base_treasures = utils.hashabledict()
        base_treasures["base"] = base
        for key in treasure_dict:
            base_treasures[key] = treasure_dict[key]["location"]
        return base_treasures

    def h(self, state, possible_actions, base_flag, treasure_dict):
        action_selected = None
        for action in possible_actions:
            if action[0][0] == "collect":
                action_selected = action
            elif action_selected == None and action[0][0] == "deposit":
                return action
        if action_selected is not None:
            return action_selected
        if action_selected is None:
            if treasure_dict == False:
                bfs_key = "base"
            else:
                bfs_key = self.base_treasures_dict(treasure_dict, base_flag)
            best_score = -1
            for action in possible_actions:
                if action[0][0] == "sail":
                    score_from_action = self.bfs_maps[bfs_key][action[0][2][0]][action[0][2][1]]
                    new_loc = action[0][2]
                    #check possible collision with marine
                    if new_loc in self.marines_next_locs(state):
                        score_from_action -= (0.5 + 0*4*(2-state["pirate_ships"][self.pirate_for_example]["capacity"]))

                    if score_from_action > best_score:
                        best_score = score_from_action
                        action_selected = action
        return action_selected

    def marines_next_locs(self, state):
        marine_ships = state["marine_ships"]
        next_loc = []
        for marine in marine_ships:
            idx = marine_ships[marine]["index"]
            if idx > 0:
                next_loc.append(marine_ships[marine]["path"][idx - 1])
            if idx < len(marine_ships[marine]["path"]) - 1:
                next_loc.append(marine_ships[marine]["path"][idx + 1])
            next_loc.append(marine_ships[marine]["path"][idx])
        return next_loc


#-----------------INFINITY----------------------------
class InfinitePirateAgent:

    def policy_find(self):
        prev_vals = {json.dumps(state): 0 for state in self.states}
        cur_vals = {json.dumps(state): 0 for state in self.states}
        state_actions_map = {json.dumps(state): self.actions_for_state(state) for state in self.states}

        state_action_probabilities = {json.dumps(state): {action: [] for action in self.actions_for_state(state)}
                                      for state in self.states}

        for state in self.states:
            for action in self.actions_for_state(state):
                state_action_probabilities[json.dumps(state)][action] = self.state_prob_score(state, action)


        optimal_strategy = {json.dumps(state): 0 for state in self.states}
        is_converged = False



        while not is_converged:
            is_converged = True
            for state in self.states:
                # find max value from possible actions
                max_value_from_state = float('-inf')
                optimal_action = None
                actions_from_state = state_actions_map[json.dumps(state)]
                for action in actions_from_state:
                    action_val = 0
                    for s_prime, p, score_from_action in state_action_probabilities[json.dumps(state)][action]:
                        action_val += p * (self.gamma*(prev_vals[json.dumps(s_prime)]) + score_from_action)
                    if action_val > max_value_from_state:
                        max_value_from_state = action_val
                        optimal_action = action
                cur_vals[json.dumps(state)] = max_value_from_state
                optimal_strategy[json.dumps(state)] = optimal_action
                # check if converged
                if abs(cur_vals[json.dumps(state)] - prev_vals[json.dumps(state)]) > 0.01:
                    is_converged = False
            # update values
            prev_vals = cur_vals.copy()


        return optimal_strategy, cur_vals

    def build_graph(self):
        """
        build the graph of the problem
        """
        n, m = len(self.map), len(
            self.map[0])
        g = nx.grid_graph((m, n))
        nodes_to_remove = []
        for node in g:
            if self.map[node[0]][node[1]] == 'I':
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            g.remove_node(node)
        return g

    def treasures_combinations(self, state, total_opts):
        treasures_dict = state['treasures']
        for t_name, t_info in treasures_dict.items():
            possible_locs = []
            for possible_location in t_info['possible_locations']:
                prob_change_location = t_info['prob_change_location'] / len(t_info['possible_locations'])
                if possible_location == t_info['location']:
                    possible = prob_change_location + (1 - t_info['prob_change_location'])
                else:
                    possible = prob_change_location
                possible_locs.append((t_name, possible_location, possible))
            total_opts.append(possible_locs)
        return total_opts

    def marine_ships_combinations(self, state, total_opts):
        marine_ships = state['marine_ships']
        for marine_name, marine in marine_ships.items():
            marine_options = []
            potential_indices = [marine['index']]
            if marine['index'] > 0:
                potential_indices.append(marine['index'] - 1)
            if marine['index'] < len(marine['path']) - 1:
                potential_indices.append(marine['index'] + 1)
            possible = 1 / len(potential_indices)
            for possible_index in potential_indices:
                marine_options.append((marine_name, possible_index, possible))
            total_opts.append(marine_options)
        return total_opts

    def final_combinations(self, total_opts):
        all_combinations = list(itertools.product(*total_opts))

        all_combos_with_probs = []
        for c in all_combinations:
            probabilities = [item[-1] for item in c]
            prod = 1
            for p in probabilities:
                prod *= p
            prod = round(prod, 6)
            all_combos_with_probs.append((prod,) + c)

        return all_combos_with_probs

    def generate_all_combinations_for_state(self, state):
        total_opts = []

        total_opts = self.treasures_combinations(state, total_opts)

        total_opts = self.marine_ships_combinations(state, total_opts)

        return self.final_combinations(total_opts)

    def marines_locations(self, state):
        marine_positions = []
        for marine in state['marine_ships'].values():
            marine_positions.append(marine['path'][marine['index']])
        return marine_positions

    def reset_board(self, state):
        copied_init = copy.deepcopy(self.initial)
        action_score_diff = -2

        marine_positions = []
        for marine in state['marine_ships'].values():
            marine_positions.append(marine['path'][marine['index']])

        for ship_name, ship in copied_init['pirate_ships'].items():

            if ship['location'] in marine_positions:
                action_score_diff -= 1
                copied_init['pirate_ships'][ship_name]['capacity'] = self.initial['pirate_ships'][ship_name]['capacity']

        return [(self.shrinked_state(copied_init), 1, action_score_diff)]

    def probs_for_atomic_action(self, state, action):
        # state_after_action = copy.deepcopy(state)
        for atomic_action in action:
            if atomic_action[0] == 'sail':
                state['pirate_ships'][atomic_action[1]]['location'] = atomic_action[2]

            if atomic_action[0] == 'collect':
                state['pirate_ships'][atomic_action[1]]['capacity'] -= 1

            if atomic_action[0] == 'deposit':
                state['pirate_ships'][atomic_action[1]]['capacity'] = \
                    self.initial['pirate_ships'][atomic_action[1]]['capacity']
        return self.generate_all_combinations_for_state(state)

    def state_prob_score(self, state, action):
        if action == "reset":
            return self.reset_board(state)

        state_after_action = copy.deepcopy(state)
        atomic_action_probs_combinations = self.probs_for_atomic_action(state_after_action, action)

        state_changes_with_probs = []
        for comb in atomic_action_probs_combinations:
            new_state = copy.deepcopy(state_after_action)
            for state_info in comb[1:]:
                if state_info[0] in self.treasures_names:
                    new_state['treasures'][state_info[0]]['location'] = state_info[1]
                else:
                    if state_info[0] in self.marines_names:
                        new_state['marine_ships'][state_info[0]]['index'] = state_info[1]



            marines_locations = self.marines_locations(new_state)

            state_points = 0

            for ship_name, ship in new_state['pirate_ships'].items():

                if ship['location'] in marines_locations:
                    state_points -= 1
                    new_state['pirate_ships'][ship_name]['capacity'] = self.initial['pirate_ships'][ship_name][
                        'capacity']

            for atomic_action in action:
                if atomic_action[0] == "deposit":
                    state_points += 4 * (self.initial['pirate_ships'][atomic_action[1]]['capacity'] -
                                         state['pirate_ships'][atomic_action[1]]['capacity'])

            state_changes_with_probs += [(self.shrinked_state(new_state), comb[0], state_points)]
        return state_changes_with_probs

    def find_policy_iteration(self):
        actions_from_all_states = {json.dumps(state): self.actions_for_state(state) for state in self.states}

        s_prime = {json.dumps(state): {action: [] for action in self.actions_for_state(state)} for state in
                      self.states}

        for state in self.states:
            for action in self.actions_for_state(state):
                s_prime[json.dumps(state)][action] = self.state_prob_score(state, action)

        # Initialize a random policy
        current_policy = {json.dumps(state): self.actions_for_state(state)[0] for state in self.states}

        optimal_value_function = {json.dumps(state): 0 for state in self.states}
        convergence_threshold = 0.01  # epsilon = 10^(−2)
        t2 = time.time()
        while t2 - self.t < 290:
            # Policy Evaluation
            while True and t2 - self.t < 290:
                delta = 0
                for state in self.states:
                    old_value = optimal_value_function[json.dumps(state)]
                    action = current_policy[json.dumps(state)]
                    new_value = 0
                    for s_tag, prob, score_change in s_prime[json.dumps(state)][action]:
                        new_value +=  prob * (self.gamma *optimal_value_function[json.dumps(s_tag)] + score_change)

                    optimal_value_function[json.dumps(state)] = new_value
                    delta = max(delta, abs(old_value - new_value))

                if delta < convergence_threshold:
                    break
            # Policy Improvement
            policy_stable = True
            for state in self.states:
                old_policy = current_policy[json.dumps(state)]

                max_V_state = float('-inf')
                best_policy = None
                actions_for_state = actions_from_all_states[json.dumps(state)]

                for action in actions_for_state:
                    value_from_action = 0
                    for s_tag, prob, score_change in s_prime[json.dumps(state)][action]:
                        value_from_action += prob * (
                                self.gamma *optimal_value_function[json.dumps(s_tag)]+ score_change)

                    if value_from_action > max_V_state:
                        max_V_state = value_from_action
                        best_policy = action

                current_policy[json.dumps(state)] = best_policy

                if old_policy != best_policy:
                    policy_stable = False

            if policy_stable:
                break

            t2 = time.time()

        return current_policy, optimal_value_function


    def all_states(self, state):
        pirate_ships = state["pirate_ships"]
        treasures = state["treasures"]
        marine_ships = state["marine_ships"]

        pirate_locations = list(self.graph.nodes())

        pirate_location_combinations = list(
            itertools.product(pirate_locations, repeat=len(pirate_ships)))  # repeat parameter is like power of __

        capacity_combinations = [range(pirate_ship["capacity"] + 1) for pirate_ship in
                                 pirate_ships.values()]

        treasure_location_combos = itertools.product(
            *[treasure["possible_locations"] for treasure in treasures.values()])

        marine_path_index_combos = itertools.product(
            *[range(len(marine_ship["path"])) for marine_ship in marine_ships.values()])

        states = []

        for pirate_combo, capacity_combo, treasure_combo, marine_index_combo in itertools.product(
                pirate_location_combinations,
                itertools.product(*capacity_combinations),
                treasure_location_combos,
                marine_path_index_combos):
            new_state = copy.deepcopy(state)

            for (location, capacity), pirate_ship_name in zip(zip(pirate_combo, capacity_combo), pirate_ships.keys()):
                new_state["pirate_ships"][pirate_ship_name]["location"] = location
                new_state["pirate_ships"][pirate_ship_name]["capacity"] = capacity

            for location, treasure_name in zip(treasure_combo, treasures.keys()):
                new_state["treasures"][treasure_name]["location"] = location

            for index, marine_ship_name in zip(marine_index_combo, marine_ships.keys()):
                new_state["marine_ships"][marine_ship_name]["index"] = index

            states.append(self.shrinked_state(new_state))
        return states

    def find_treasure_name_by_location(self, location, state):
        treasures = []
        for treasure_name, treasure_location in state['treasures'].items():
            if treasure_location['location'] == location:
                treasures.append(treasure_name)
        return treasures


    def legal_neighbors(self, x, y):
        #neighbors that are not islands
        adj = []
        if x > 0 and self.map[x - 1][y] != 'I':
            adj.append([x - 1, y])
        if x < len(self.map) - 1 and self.map[x + 1][y] != 'I':
            adj.append([x + 1, y])
        if y > 0 and self.map[x][y - 1] != 'I':
            adj.append([x, y - 1])
        if y < len(self.map) - 1 and self.map[x][y + 1] != 'I':
            adj.append([x, y + 1])
        return adj
    def actions_for_state(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        treasures_location = []
        for treasures in state['treasures'].values():
            treasures_location.append(treasures['location'])

        all_ship_actions = []
        for ship_name, ship_dict in state['pirate_ships'].items():
            pirate_ship_actions = self.actions_per_ship(ship_name, ship_dict['location'], state, treasures_location)
            all_ship_actions.append(pirate_ship_actions)

        all_possible_actions_combinations = list(itertools.product(*all_ship_actions))
        all_possible_actions_combinations += ["reset"]
        return all_possible_actions_combinations
    def actions_per_ship(self, ship, location, state, treasures_location):
        actions = []
        possible_neighbors = self.legal_neighbors(location[0], location[1])

        # deposit
        if state['pirate_ships'][ship]['location'] == self.base_location \
                and state['pirate_ships'][ship]['capacity'] < self.initial['pirate_ships'][ship]['capacity']:
            actions += [("deposit", ship)]

        # collect
        if state['pirate_ships'][ship]['capacity'] > 0:
            treasure_names = self.find_treasure_name_by_location((location[0] + 1, location[1]), state)
            for treasure_name in treasure_names:
                if location[0] + 1 < len(self.map) and any(
                        (location[0] + 1, location[1]) == coords for coords in treasures_location):
                    actions += [("collect", ship, treasure_name)]
            treasure_names = self.find_treasure_name_by_location((location[0] - 1, location[1]), state)
            for treasure_name in treasure_names:
                if location[0] - 1 >= 0 and any(
                        (location[0] - 1, location[1]) == coords for coords in treasures_location):
                    actions += [("collect", ship, treasure_name)]

            treasure_names = self.find_treasure_name_by_location((location[0], location[1] + 1), state)
            for treasure_name in treasure_names:
                if location[1] + 1 < len(self.map[0]) and any(
                        (location[0], location[1] + 1) == coords for coords in treasures_location):
                    actions += [("collect", ship, treasure_name)]

            treasure_names = self.find_treasure_name_by_location((location[0], location[1] - 1), state)
            for treasure_name in treasure_names:
                if location[1] - 1 >= 0 and any(
                        (location[0], location[1] - 1) == coords for coords in treasures_location):
                    actions += [("collect", ship, treasure_name)]

        # sail
        for neighbor in possible_neighbors:
            actions += [("sail", ship, (neighbor[0], neighbor[1]))]

        actions += [("wait", ship)]
        return actions

    def shrinked_state(self, original_dict):
        keys = ["pirate_ships", "treasures", "marine_ships"]
        new_dict = {key: original_dict[key] for key in keys if key in original_dict}
        return new_dict

    def __init__(self, initial, gamma):
        self.t = time.time()
        self.initial = initial
        self.gamma = gamma
        self.treasures_names = list(initial["treasures"].keys())
        self.marines_names = list(initial["marine_ships"].keys())
        self.base_location = next(iter(initial['pirate_ships'].values()), None).get("location")
        self.map = initial['map']

        self.graph = self.build_graph()
        self.states = self.all_states(self.initial)
        self.policy, self.optimal_value_function = self.policy_find()

    def act(self, state):
        state = self.shrinked_state(state)
        return self.policy[json.dumps(state)]

    def value(self, state):
        """implement the value(self, state) function which returns 𝑉∗(𝑠𝑡𝑎𝑡𝑒)"""
        #if the discount factor is from state 1 - return this
        return self.optimal_value_function[json.dumps(self.shrinked_state(state))]

        #if the discount factor is not from state 1 - return this
        #return self.optimal_value_function[json.dumps(self.shrinked_state(state))] / self.gamma


def state_to_dict(state):
    """
    converts state to dictionary
    """
    return json.loads(state)


def dict_to_state(dict):
    """
    converts dictionary to state (json representation)
    """
    return json.dumps(dict, sort_keys=True)


def shrinked_state(original_dict):
    keys = ["pirate_ships", "treasures", "marine_ships"]
    new_dict = {key: original_dict[key] for key in keys if key in original_dict}
    return new_dict


