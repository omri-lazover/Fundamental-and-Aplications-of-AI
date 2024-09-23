import json
import time
from itertools import product

import search_318862323_206373367
import random
import math

from utils import hashabledict, norm, removeall, Queue

infinity = float("inf")
ids = ["318862323", "206373367"]


class OnePieceProblem(search_318862323_206373367.Problem):
    """This class implements a medical problem according to problem description file"""

    # ________________________INITIAL_____________________________________________

    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""
        self.map = initial.pop("map")
        pirate_ships = initial["pirate_ships"]
        self.treasures = initial["treasures"]
        marine_ships = initial["marine_ships"]

        # find base
        for k, v in pirate_ships.items():
            self.base = v
            break

        self.ships = initial["pirate_ships"].keys()
        self.treasures_names = initial["treasures"].keys()
        self.treasures_locs = initial["treasures"].values()

        self.dist_mat = self.bfs_distance_to_base(self.map)
        self.dist_mat2 = self.bfs_all_distances(self.map)

        # updating the pirate ships info
        for k, v in pirate_ships.items():
            pirate_ships[k] = {"location": v, "treasures": []}

        # updating the marine ships info
        for k, v in marine_ships.items():
            marine_ships[k] = {"track": v + v[::-1][1:-1], "location index": 0}

        # updating the treasures info
        for k, v in initial["treasures"].items():
            initial["treasures"][k] = {"island": v, "on ships": [], "deposited": False}

        self.distances = {}

        for name, dict in initial["treasures"].items():
            self.distances[name] = self.sssp(dict["island"])

        self.sssp_from_base = self.sssp(self.base)

        initial["treasures_deposited"] = []

        search_318862323_206373367.Problem.__init__(self, self.to_state(initial))

    def to_state(self, dictionary):
        if dictionary is not None:
            return json.dumps(dictionary, sort_keys=True)
        else:
            # Handle the case when dictionary is None (return a default value or raise an exception)
            # For example, you can return an empty JSON string:
            return '{}'

    def to_dict(self, state):
        return json.loads(state)

    def neighbors(self, x, y):
        adj = []
        if x > 0:
            adj.append([x - 1, y])
        if x < len(self.map) - 1:
            adj.append([x + 1, y])
        if y > 0:
            adj.append([x, y - 1])
        if y < len(self.map) - 1:
            adj.append([x, y + 1])
        return adj

    # _________________ACTIONS__________________________________________________________

    def collected(self, ships, treasure):
        for k, v in ships.items():
            if treasure in v["treasures"]:
                return True
        return False

    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        dict = self.to_dict(state)
        pirates = dict["pirate_ships"]
        marines = dict["marine_ships"]
        acts = []
        for name, info in pirates.items():
            acts.append(self.action_per_pirate(name, info, dict["treasures"], pirates, marines))

        return list(product(*acts))

    def action_per_pirate(self, p, pirate_state, treasures, pirates, marines):
        x, y = pirate_state["location"]

        marines_next_locs = self.marines_locations_after_turn(marines)
        # if (x, y) not in marines_next_locs:
        possible_actions = [("wait", p)]

        """___________POSSIBLE MOVES_________________________________"""
        if x > 0 and self.map[x - 1][y] in ['S', 'B'] \
                and ((x - 1, y) not in marines_next_locs or pirate_state["treasures"] == []):  # my change
            possible_actions.append(("sail", p, (x - 1, y)))
        if x < len(self.map) - 1 and self.map[x + 1][y] in ['S', 'B'] \
                and ((x + 1, y) not in marines_next_locs or pirate_state["treasures"] == []):  # my change
            possible_actions.append(("sail", p, (x + 1, y)))
        if y > 0 and self.map[x][y - 1] in ['S', 'B'] \
                and ((x, y - 1) not in marines_next_locs or pirate_state["treasures"] == []):  # my change
            possible_actions.append(("sail", p, (x, y - 1)))
        if y < len(self.map[0]) - 1 and self.map[x][y + 1] in ['S', 'B'] \
                and ((x, y + 1) not in marines_next_locs or pirate_state["treasures"] == []):  # my change
            possible_actions.append(("sail", p, (x, y + 1)))

        """_________POSSIBLE COLLECTIONS______________________________________"""
        if len(pirate_state["treasures"]) < 2:
            adj_tiles = self.neighbors(x, y)
            for treasure_name, treasure_info in treasures.items():
                if treasure_info["island"] in adj_tiles and treasure_info["deposited"] == False \
                        and treasure_name not in pirate_state["treasures"] and self.collected(pirates,
                                                                                              treasure_name) == False:

                    try:
                        check = not (treasure_name == min(treasures.keys()) and p != max(pirates.keys()))
                    except ValueError:
                        check = True

                    # if not (treasure_name == min(treasures.keys()) and p!= max(pirates.keys())): #my change
                    if check:
                        if (x, y) not in marines_next_locs:  # my change
                            possible_actions.append(("collect_treasure", p, treasure_name))

        """_________POSSIBLE DEPOSIT____________________________________"""
        if self.map[x][y] == 'B' \
                and len(pirate_state["treasures"]) > 0:
            possible_actions.append(("deposit_treasures", p))

        return possible_actions

    def marines_locations_after_turn(self, marines):
        locs = []
        for marine in marines.values():
            track = marine["track"]
            track_len = len(track)
            idx = marine["location index"]
            next_loc_idx = (idx + 1) % track_len
            locs.append(track[next_loc_idx])
        return locs

    # __________________RESULT________________________________________________________________

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        new_state = self.result_per_pirate(state, action)
        new_state2 = self.update_marines(new_state)
        new_state3 = self.update_treasures(new_state2)

        """if (action[0][0] in ["deposit_treasures"]):
            pass
            print("state: ", state)
            print("action: ", action)

            print("new state: ", new_state3)
            print()
            print()
            print()"""

        return new_state3

    def result_per_pirate(self, state, action):

        dict = self.to_dict(state)

        for act in action:
            a = act[0]  # sail/deposit/wait/collect
            name = act[1]  # PIRATE SHIP NAME

            pirate_state = dict["pirate_ships"][name]
            treasures = dict["treasures"]  # treasures

            if a == "sail":
                location = act[2]
                pirate_state["location"] = location

            elif a == "wait":
                pass

            elif a == "collect_treasure":
                t = act[2]
                pirate_state["treasures"].append(t)

                treasures[t]["on ships"].append(name)

                # My changes
                pirate_state["treasures"].sort()
                treasures[t]["on ships"].sort()

            elif a == "deposit_treasures":

                for t in pirate_state["treasures"]:
                    if t not in dict["treasures_deposited"]:
                        dict["treasures_deposited"].append(t)
                        treasures[t]["deposited"] = True
                        dict["treasures"][t]["on ships"] = [x for x in dict["treasures"][t]["on ships"] if x != name]
                pirate_state["treasures"] = []

                # my change
                dict["treasures_deposited"].sort()

        return self.to_state(dict)

    def update_marines(self, state):
        dict = self.to_dict(state)
        marines = dict["marine_ships"]
        for k, v in marines.items():
            track = v["track"]
            track_len = len(track)
            idx = v["location index"]
            v["location index"] = (idx + 1) % track_len
        return self.to_state(dict)

    def update_treasures(self, state):
        dict = self.to_dict(state)
        marines = dict["marine_ships"]
        marines_locations = []

        # list of marines locations
        for k, v in marines.items():
            track = v["track"]
            idx = v["location index"]
            marines_locations.append(track[idx])

        p_ships = dict["pirate_ships"]
        for k, v in p_ships.items():
            loc = v["location"]
            if loc in marines_locations and len(
                    v["treasures"]) > 0:  # if a pirate ship steps on a tile with a marine ship
                # print("check")
                for tr in v["treasures"]:
                    dict["treasures"][tr]["on ships"] = [x for x in dict["treasures"][tr]["on ships"] if
                                                         x != k]  # removeall(dict["treasures"][tr]["on ships"], k)
                v["treasures"] = []  # forfeit treasures

        return self.to_state(dict)

    # ________________END RESULT________________________________________________________________

    # __________________GOAL TEST________________________________________________________________

    def goal_test(self, state):
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""
        st = self.to_dict(state)
        return len(st["treasures_deposited"]) == len(self.treasures_names)

    # __________________ END GOAL TEST________________________________________________________________

    def h(self, node):
        dict = self.to_dict(node.state)

        if len(dict["pirate_ships"]) > 2:
            return (self.my_h(node))
        else:
            return max(self.my_h2(node), self.h_1(node))

    # not admissible
    def my_h(self, node):
        dict = self.to_dict(node.state)
        pirate_ships = dict["pirate_ships"]
        treasures = dict["treasures"]
        marine_ships = dict["marine_ships"]

        total_distance = 0
        num_pirates = len(pirate_ships)
        xb, yb = self.base
        used_treasures = []

        for k, v in pirate_ships.items():

            xs, ys = v["location"]

            if len(v["treasures"]) == 2:
                # add distance to base
                total_distance += self.sssp_from_base[xs][ys]


            elif len(v["treasures"]) == 1:

                sorted_dist = self.sort_treasures(v,
                                                  treasures)  # sorted list of the uncollected treasures by distance to the ship
                allocated = False

                for t_name, distance in sorted_dist:

                    xi, yi = treasures[t_name]["island"]
                    t_dict = treasures[t_name]

                    island_neighbors = [(x, y) for x, y in self.neighbors(xi, yi) if self.map[x][y] in ['S', 'B']]
                    closest_ship, distance = self.find_closest_ship(t_name, pirate_ships, treasures)

                    if t_name not in used_treasures and closest_ship == k:
                        used_treasures.append(t_name)

                        total_distance += distance
                        allocated = True
                        break
                if allocated == False:
                    total_distance += self.sssp_from_base[xs][ys]



            else:
                t_name, dist = self.find_closest_treasure(v, treasures, used_treasures)

                if t_name == None:
                    continue
                xi, yi = treasures[t_name]["island"]
                t_dict = treasures[t_name]
                island_neighbors = [(x, y) for x, y in self.neighbors(xi, yi) if self.map[x][y] in ['S', 'B']]
                used_treasures.append(t_name)
                total_distance += dist

        # for unallocated treasures:

        uncollected = self.uncollected_treasures(treasures)
        for t_name in uncollected:
            xi, yi = treasures[t_name]["island"]
            t_dict = treasures[t_name]
            island_neighbors = [(x, y) for x, y in self.neighbors(xi, yi) if self.map[x][y] in ['S', 'B']]
            if t_name is not None and t_name not in used_treasures:
                distance_from_neighbors_to_base = [self.sssp_from_base[x][y] for x, y in island_neighbors]
                total_distance += 2 * min(distance_from_neighbors_to_base, default=0)

        return total_distance

    def my_h2(self, node):
        dict = self.to_dict(node.state)
        pirate_ships = dict["pirate_ships"]
        treasures = dict["treasures"]
        marine_ships = dict["marine_ships"]

        total_distance = 0
        num_pirates = len(pirate_ships)

        for k, v in treasures.items():
            if v["deposited"]:
                total_distance += 0
            elif v["on ships"]:
                # Calculate distances to treasures on ships and islands
                ships_loc = []
                for ship in v["on ships"]:
                    x, y = pirate_ships[ship]["location"]
                    ships_loc.append(self.sssp_from_base[x][y])
                if ships_loc:
                    total_distance += min(ships_loc)
            else:
                min_d = float("inf")
                # Calculate distances to treasures on islands
                xi, yi = v["island"]
                xb, yb = self.base
                possible_dists = []
                adj = self.neighbors(xi, yi)
                for ship in pirate_ships.values():
                    xs, ys = ship["location"]
                    dist_to_island = self.distances[k][xi][yi] + self.distances[k][xb][yb]
                    min_d = min(min_d, dist_to_island)
                total_distance += min_d

        return total_distance / num_pirates

    def find_optimal_allocation(self, ships, treasures, distance_matrix):

        # Step 1: Subtract the minimum value from each row
        for i in range(len(distance_matrix)):
            min_val = min(distance_matrix[i])
            for j in range(len(distance_matrix[i])):
                distance_matrix[i][j] -= min_val

        # Step 2: Subtract the minimum value from each column
        for j in range(len(distance_matrix[0])):
            min_val = min(distance_matrix[i][j] for i in range(len(distance_matrix)))
            for i in range(len(distance_matrix)):
                distance_matrix[i][j] -= min_val

        # Step 3: Mark zeros to cover all zeros with the minimum number of lines
        rows_covered = set()
        cols_covered = set()

        while True:
            for i in range(len(distance_matrix)):
                for j in range(len(distance_matrix[i])):
                    if distance_matrix[i][j] == 0 and i not in rows_covered and j not in cols_covered:
                        rows_covered.add(i)
                        cols_covered.add(j)

            if len(rows_covered) + len(cols_covered) >= len(distance_matrix):
                break

            # Find the smallest uncovered value
            min_uncovered = float('inf')
            for i in range(len(distance_matrix)):
                for j in range(len(distance_matrix[i])):
                    if i not in rows_covered and j not in cols_covered:
                        min_uncovered = min(min_uncovered, distance_matrix[i][j])

            # Subtract the smallest uncovered value from all uncovered elements
            for i in range(len(distance_matrix)):
                for j in range(len(distance_matrix[i])):
                    if i not in rows_covered and j not in cols_covered:
                        distance_matrix[i][j] -= min_uncovered

        # Step 4: Find the minimum number of lines to cover all zeros
        lines_count = len(rows_covered) + len(cols_covered)

        # If the number of lines is equal to the matrix size, it's an optimal solution
        if lines_count == len(distance_matrix):
            optimal_allocation = [(ships[i], treasures[j]) for i, j in enumerate(cols_covered)]
            return optimal_allocation

    def sort_treasures(self, pirate_dict, treasures):
        xp, yp = pirate_dict["location"]
        closest_treasure_name = None
        distance_array = []
        for t_name, t_dict in treasures.items():
            if t_dict["deposited"] == False and not t_dict["on ships"]:
                xi, yi = treasures[t_name]["island"]
                closest_neighbor_distance = min((self.dist_mat2[xp][yp][x][y]  # one option
                                                 + self.sssp_from_base[x][y]
                                                 for x, y in self.neighbors(xi, yi) if self.map[x][y] in ['S', 'B']),
                                                default=float("inf"))
                distance_array.append((t_name, closest_neighbor_distance))

        sorted_treasures = sorted(distance_array, key=lambda x: x[1])

        return sorted_treasures

    def find_closest_treasure(self, pirate_dict, treasures, used_treasures):
        min_d = float("inf")
        xp, yp = pirate_dict["location"]
        xb, yb = self.base
        closest_treasure_name = None
        for t_name in self.uncollected_treasures(treasures):
            t_dict = treasures[t_name]
            xi, yi = t_dict["island"]
            if t_name not in used_treasures:
                dist_to_treasure = min((self.dist_mat2[xp][yp][x][y]
                                        + self.dist_mat2[x][y][xb][yb]
                                        for x, y in self.neighbors(xi, yi) if self.map[x][y] in ['S', 'B']),
                                       default=float("inf"))

                if dist_to_treasure < min_d:
                    min_d = dist_to_treasure
                    closest_treasure_name = t_name
        return (closest_treasure_name, min_d)

    def find_closest_ship(self, treasure_name, pirates, treasures):
        min_d = float("inf")
        xb, yb = self.base
        closest_ship_name = None
        xi, yi = treasures[treasure_name]["island"]
        for s_name, s_dict in pirates.items():
            if len(s_dict["treasures"]) < 2:
                xp, yp = s_dict["location"]
                """dist_to_ship =min((self.dist_mat2[xp][yp][x][y] + self.dist_mat2[x][y][xb][yb]
                                  for x, y in self.neighbors(xi, yi) if self.map[x][y] in ['S', 'B']), default=float("inf"))"""
                dist_to_ship = self.find_sea_adj_for_h(xi, yi, xp, yp)
                if dist_to_ship < min_d:
                    min_d = dist_to_ship
                    closest_ship_name = s_name
        return (closest_ship_name, min_d)

    def find_sea_adj_for_h(self, cord_x, cord_y, x_cord_pirate, y_cord_pirate):
        possible_moves = [(x, y) for x, y in self.neighbors(cord_x, cord_y) if self.map[x][y] in ['B', 'S']]
        my_sea_adj_distances = []
        for move in possible_moves:
            x, y = move

            my_sea_adj_distances.append(self.sssp_from_base[x][y] + self.dist_mat2[x_cord_pirate][y_cord_pirate][x][y])

        return min(my_sea_adj_distances)

    def distance_to_closest_island_neighbor(self, source, island):
        x_source, y_source = source
        x_island, y_island = island
        valid_island_neighbors = [(x, y) for x, y in self.neighbors(x_island, y_island) if self.map[x][y] in ['S', 'B']]

        distances_to_neighbors = [self.sssp_from_base[x][y] for (x, y) in valid_island_neighbors]
        return min(distances_to_neighbors, default=infinity)

    def shortest_path_to_base_from_neighbors(self, x, y):
        neighbors = [(row, col) for row, col in self.neighbors(x, y) if self.map[row][col] in ['S', 'B']]

        if not neighbors:
            # No valid neighbors, return a default value for infinity
            return float('inf'), None
        # Find the neighbor with the shortest path to the base
        shortest_neighbor = min(neighbors,
                                key=lambda neighbor: self.sssp_from_base[neighbor[0]][neighbor[1]])

        # Return the shortest path distance and the neighbor
        shortest_distance = self.sssp_from_base[shortest_neighbor[0]][shortest_neighbor[1]]
        return shortest_distance, shortest_neighbor

    # ----------------H1--------------------------
    def all_on_island(self, dict):
        for k, v in dict["treasures"].items():
            if v["deposited"] == True or v["on ships"] != []:
                return False
        return True

    def h_1(self, node):
        state = self.to_dict(node.state)
        num_pirates = len(state["pirate_ships"])

        count = 0
        for k, v in state["treasures"].items():
            if v["on ships"] == [] and v["deposited"] == False:
                count += 1

        return count / num_pirates

    # ----------------H2--------------------------------
    def bfs_distance_to_base(self, grid):

        # Finding the source to start from
        x_source, y_source = self.base

        # To maintain location visit status
        tiles_visited = [[False] * len(grid[0]) for _ in range(len(grid))]  # map of Falses
        distances_matrix = [[-1 for _ in range(len(grid[0]))] for _ in range(len(grid))]  # map of -1s

        source = (x_source, y_source, 0)  # (x, y, distance)

        # applying BFS on matrix cells starting from source
        queue = []
        queue.append(source)
        tiles_visited[x_source][y_source] = True
        distances_matrix[x_source][y_source] = 0  # Distance from 'B' to itself is 0

        while len(queue) > 0:
            tile = queue.pop(0)
            row = tile[0]
            col = tile[1]
            dist = tile[2]
            # moving left
            if row > 0 and (grid[row - 1][col] == 'S') and (tiles_visited[row - 1][col] == False):
                queue.append((row - 1, col, dist + 1))
                tiles_visited[row - 1][col] = True
                distances_matrix[row - 1][col] = dist + 1

            # moving right
            if row < len(grid) - 1 and (grid[row + 1][col] == 'S') and (tiles_visited[row + 1][col] == False):
                queue.append((row + 1, col, dist + 1))
                tiles_visited[row + 1][col] = True
                distances_matrix[row + 1][col] = dist + 1

            if col > 0 and (grid[row][col - 1] == 'S') and (tiles_visited[row][col - 1] == False):
                queue.append((row, col - 1, dist + 1))
                tiles_visited[row][col - 1] = True
                distances_matrix[row][col - 1] = dist + 1

            if col < len(grid[0]) - 1 and (grid[row][col + 1] == 'S') and (tiles_visited[row][col + 1] == False):
                queue.append((row, col + 1, dist + 1))
                tiles_visited[row][col + 1] = True
                distances_matrix[row][col + 1] = dist + 1

        return distances_matrix

    def h_2(self, node):
        dict = self.to_dict(node.state)
        xb, yb = self.base
        total_dist = 0
        treasures = dict["treasures"]
        for t_name, t_dict in treasures.items():
            if t_dict["deposited"] == True:
                total_dist += 0
            elif t_dict["on ships"] == []:
                x, y = t_dict["island"]
                adj = [(x1, y1) for x1, y1 in self.neighbors(x, y) if self.map[x1][y1] == 'S']
                if not adj:
                    return float("inf")
                total_dist += min([(abs(x - xb) + abs(y - yb)) for x, y in adj])
            else:
                min_dist = float('inf')
                for s in t_dict["on ships"]:
                    x, y = dict["pirate_ships"][s]["location"]
                    adj = [(x1, y1) for x1, y1 in self.neighbors(x, y) if self.map[x1][y1] == 'S']
                    if not adj:
                        return float("inf")
                    total_dist_neighbors = [(abs(x - xb) + abs(y - yb)) for x, y in adj]
                    min_dist = min(min_dist, min(total_dist_neighbors))
                total_dist += min_dist
        ships_num = len(dict["pirate_ships"])
        return total_dist / ships_num

    # ----------MY HEURISTICS-----------------------

    def step_on_marine(self, dict):
        marines = dict["marine_ships"]
        marines_locations = []
        for k, v in marines.items():
            idx = v["location index"]
            track = v["track"]
            x, y = track[idx]
            marines_locations.append((x, y))
        pirates = dict["pirate_ships"]
        for k, v in pirates.items():
            x, y = v["location"]
            if (x, y) in marines_locations:
                return True
        return False

    def h4(self, node):
        dict = self.to_dict(node.state)
        pirate_ships = dict["pirate_ships"]
        treasures = dict["treasures"]
        marine_ships = dict["marine_ships"]

        total_distance = 0
        num_pirates = len(pirate_ships)

        for k, v in treasures.items():
            if v["deposited"]:
                total_distance += 0
            elif v["on ships"]:
                # Calculate distances to treasures on ships and islands
                adj_to_ships = []
                for ship in v["on ships"]:
                    x, y = pirate_ships[ship]["location"]
                    adj_to_ships += [self.dist_mat[adj[0]][adj[1]] for adj in self.neighbors(x, y) if
                                     self.map[adj[0]][adj[1]] in ['S', 'B']]
                if adj_to_ships:
                    total_distance += min(adj_to_ships)
            else:
                # Calculate distances to treasures on islands
                x, y = v["island"]
                adj = self.neighbors(x, y)
                dist_to_island = min(
                    (self.dist_mat[tile[0]][tile[1]] for tile in adj if self.map[tile[0]][tile[1]] in ['S', 'B']),
                    default=float('inf'))
                total_distance += dist_to_island

        """ # Consider marine ships in the distance calculation
        PENALTY_VALUE = 100

        for pirate_ship in pirate_ships.values():
            if pirate_ship["treasures"]:
                x_pirate, y_pirate = pirate_ship["location"]
                for marine_ship in marine_ships.values():
                    x_marine, y_marine = marine_ship["track"][marine_ship["location index"]]
                    if x_pirate == x_marine and y_pirate == y_marine:
                        # Penalize the pirate ship for being on the same tile as a marine ship
                        total_distance += PENALTY_VALUE
                        break  # Break the loop since we penalized the pirate ship once
"""
        return total_distance / num_pirates

    def bfs_all_distances(self, grid):
        rows, cols = len(grid), len(grid[0])
        distances_matrix = [
            [
                [
                    [-1 for _ in range(cols)]
                    for _ in range(rows)
                ]
                for _ in range(cols)
            ]
            for _ in range(rows)
        ]

        for start_row in range(rows):
            for start_col in range(cols):
                if grid[start_row][start_col] in ['S', 'B']:
                    queue = [(start_row, start_col, 0)]

                    visited = [[False] * cols for _ in range(rows)]
                    visited[start_row][start_col] = True

                    while queue:
                        current_row, current_col, distance = queue.pop(0)
                        distances_matrix[start_row][start_col][current_row][current_col] = distance

                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            new_row, new_col = current_row + dr, current_col + dc
                            if 0 <= new_row < rows and 0 <= new_col < cols and grid[new_row][new_col] in ['S',
                                                                                                          'B'] and not \
                                    visited[new_row][new_col]:
                                visited[new_row][new_col] = True
                                queue.append((new_row, new_col, distance + 1))

        return distances_matrix

    def sssp(self, source):
        grid = self.map
        # Finding the source to start from
        x_source, y_source = source

        # To maintain location visit status
        tiles_visited = [[False] * len(grid[0]) for _ in range(len(grid))]  # map of Falses
        distances_matrix = [[float("inf") for _ in range(len(grid[0]))] for _ in range(len(grid))]  # map of -1s

        source = (x_source, y_source, 0)  # (x, y, distance)

        # applying BFS on matrix cells starting from source
        queue = []
        queue.append(source)
        tiles_visited[x_source][y_source] = True
        distances_matrix[x_source][y_source] = 0  # Distance from 'B' to itself is 0

        while len(queue) > 0:
            tile = queue.pop(0)
            row = tile[0]
            col = tile[1]
            dist = tile[2]

            if row > 0 and (grid[row - 1][col] in ['S', 'B']) and (tiles_visited[row - 1][col] == False):
                queue.append((row - 1, col, dist + 1))
                tiles_visited[row - 1][col] = True
                distances_matrix[row - 1][col] = dist + 1

            if row < len(grid) - 1 and (grid[row + 1][col] in ['S', 'B']) and (tiles_visited[row + 1][col] == False):
                queue.append((row + 1, col, dist + 1))
                tiles_visited[row + 1][col] = True
                distances_matrix[row + 1][col] = dist + 1

            if col > 0 and (grid[row][col - 1] in ['S', 'B']) and (tiles_visited[row][col - 1] == False):
                queue.append((row, col - 1, dist + 1))
                tiles_visited[row][col - 1] = True
                distances_matrix[row][col - 1] = dist + 1

            if col < len(grid[0]) - 1 and (grid[row][col + 1] in ['S', 'B']) and (tiles_visited[row][col + 1] == False):
                queue.append((row, col + 1, dist + 1))
                tiles_visited[row][col + 1] = True
                distances_matrix[row][col + 1] = dist + 1

        return distances_matrix

    def get_num_pirates(self, node):
        state = node.state
        dict = self.to_dict(state)

        num_pirates = len(dict['pirate_ships'])

        return num_pirates

    def uncollected_treasures(self, treasures):
        l = []
        for k, v in treasures.items():
            if v["deposited"] == False and not v["on ships"]:
                l.append(k)
        return l


def create_onepiece_problem(game):
    return OnePieceProblem(game)

