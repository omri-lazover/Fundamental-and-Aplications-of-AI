import math
from abc import abstractmethod, ABC
from queue import Queue, PriorityQueue, LifoQueue
from mpmath import memoize


class Problem(ABC):
    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly
        a goal state, if there is a unique goal. Your subclass’s
        constructor can add other arguments."""
        self.initial = initial;
        self.goal = goal

    @abstractmethod
    def actions(self, state):
        """Return the actions that can be executed in the given
        state."""
        pass
        #abstract

    @abstractmethod
    def result(self, state): #should add an action???
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        pass
        #abstract

    def goal_test(self, state):
        """Return True if the state is a goal. The default
        method compares the state to self.goal, as specified in the
        constructor. Implement this method if checking against a
        single self.goal is not enough."""

        #for single goal state:
        return state == self.goal

        #for more than 1 goal states:
        #return state in self.goal

    def path_cost(self, c, state1, action, state2): #action = state1->state2

        """Return the cost of a solution path that arrives at state2
        from state1 via action, assuming cost c to get up to state1.
        If the problem is such that the path doesn’t matter, this
        function will only look at state2. If the path does matter,
        it will consider c and maybe state1 and action. The default
        method costs 1 for every step in the path."""
        return c + 1

class Node:

    def __init__(self, state, parent=None, action=None, path_cost = 0):
        """Create a search tree Node, derived from a parent by an action."""

        #update(self, state=state, parent=parent, action=action, path_cost = path_cost, depth = 0)
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0

        if parent:
            self.depth = parent.depth + 1



    def __repr__(self):
        return "<Node %s>" % (self.state,)

    def path(self):
        """Create a list of nodes from the root to this node."""

        x, result = self, [self]
        while x.parent:
            result.append(x.parent)
            x = x.parent
        return result

    def expand(self, problem):
        "Return a list of nodes reachable from this node."

        return [self.child_node(problem, action)
        for action in
        problem.actions(self.state)]


    def child_node(self, problem, action):

        next = problem.result(self.state, action)
        return Node(next, self, action, problem.path_cost(self.path_cost, self.state, action, next))



"""_________SEARCH FUNCTIONS____________________________________"""

def graph_search(problem, fringe):
    """Search through the successors of a problem to find a goal.
    The argument fringe should be an empty queue.
    If two paths reach a state, only use the best one."""
    closed = {} # ← CLOSED list
    fringe.put(Node(problem.initial))
    while fringe.qsize()>0:
        node = fringe.get()
        if problem.goal_test(node.state):
            return node
        if node.state not in closed:
            closed[node.state] = True

            #fringe.extend(node.expand(problem))
            for element in node.expand(problem): #replace the upper line
                fringe.put(element)

    return None

def breadth_first_graph_search(problem):
    "Search the shallowest nodes in the search tree first."
    return graph_search(problem, Queue())

def depth_first_graph_search(problem):
    "Search the deepest nodes in the search tree first."
    return graph_search(problem, LifoQueue())

def best_first_graph_search(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize;
    for example, if f is a heuristic estimate to the goal, then we
    have greedy best first search; if f is node.depth then we have
    breadth-first search. There is a subtlety: the line "f =
    memoize(f, ’f’)" means that the f values will be cached on the
    nodes as they are computed. So after doing a best first search
    you can examine the f values of the path returned."""
    f = memoize(f, "f")
    return graph_search(problem, PriorityQueue(min, f))


def astar_search(problem, h=None):
    """A* search is best-first graph search with f(n)=g(n)+h(n).
    You need to specify the h function when you call astar search.
    Uses the pathmax trick: f(n) = max(f(n), g(n) + h(n))."""
    h = h or problem.h
    def f(n):
        return max(getattr(n, "f", float("-inf")), n.path_cost + h(n))
    return best_first_graph_search(problem, f)

"""________________________________________________________________________"""



class MazeProblem(Problem):
    def __init__(self, initial, goal, maze):
        super().__init__(initial, goal)
        self.maze = maze

    def h(self, node: Node):
        x, y = node.state
        x2, y2 = self.goal.state
        return math.sqrt((x-x2)**2 + (y-y2)**2)

    def actions(self, state):
        x, y = state
        possible_actions = []
        if x > 0 and self.maze[x - 1][y] == '#':
            possible_actions.append('left')
        if x < len(self.maze) - 1 and self.maze[x + 1][y] == '#':
            possible_actions.append('right')
        if y > 0 and self.maze[x][y - 1] == '#':
            possible_actions.append('up')
        if y < len(self.maze[0]) - 1 and self.maze[x][y + 1] == '#':
            possible_actions.append('down')
        return possible_actions

    def result(self, state, action):
        x, y = state
        if action == 'left':
            return x - 1, y
        elif action == 'right':
            return x + 1, y
        elif action == 'up':
            return x, y - 1
        elif action == 'down':
            return x, y + 1
        else:
            return state

    def path_cost(self, c, state1, action, state2):
        return c + 1  # Uniform cost for simplicity



def main():
    # Example maze
    maze = [
        "S#####",
        "#     ",
        "# ### ",
        "#   # ",
        "### # ",
        "######"
    ]

    maze2 = [
        "S#####",
        "      ",
        "# ### ",
        "#   # ",
        "### # ",
        "######"
    ]

    initial_state = (0, 0)  # Starting position
    goal_state = (5, 1)  # Goal position

    maze_problem = MazeProblem(initial_state, goal_state, maze)
    #maze_problem = MazeProblem(initial_state, goal_state, maze2)

    # Run breadth-first graph search
    result_node = breadth_first_graph_search(maze_problem)

    # Print the solution path
    if result_node:
        solution_path = result_node.path()
        for step, node in enumerate(solution_path):
            print(f"Step {step}: {node.state}")
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()



