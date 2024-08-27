# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import heapq

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """
    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]

def getStateId(state):
    return state

def genericSearch(problem, fringe, allow_revisiting = False):

    visited = set()
    totalPath = list()
    fringe.push((problem.getStartState(), list(), 0))
    while not fringe.isEmpty():
        currentState = fringe.pop()
        if problem.isGoalState(currentState[0]) == True:
            return currentState[1]
        if getStateId(currentState[0]) not in visited or allow_revisiting:
            for childNode, action, childCost in problem.getSuccessors(currentState[0]):
                    totalPath = currentState[1].copy()
                    totalPath.append(action)
                    totalCost = currentState[2] + childCost
                    fringe.push((childNode, totalPath, totalCost))
        visited.add(getStateId(currentState[0]))

    return None


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    
    return genericSearch(problem, util.Stack())

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    
    return genericSearch(problem, util.Queue())

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    
    return genericSearch(problem, util.PriorityQueueWithFunction(lambda x: x[2]))
    

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    
    class aStarFringe(util.PriorityQueueWithFunction):
        """
        Implements the fringe data structure required for A star search
        using a priority queue and two dictionaries - open set and closet set
        """
        def  __init__(self, problem, heuristic):
            self.prob = problem
            self.g_func = lambda x: x[2] # returns g value, third value in the node tuple is the total cost
            self.h_func = lambda x: heuristic(x[0], self.prob) # returns h value, estimate to the goal node
            self.f_func = lambda x: self.g_func(x) + self.h_func(x) # returns f value, f = g + h
            self.open_set, self.closed_set = {}, {}
            super().__init__(lambda x: self.f_func(x))        # super-class initializer

        def update(self, item, priority, item_id = lambda x: x):
            # util copied from util.py
            # If item already in priority queue with higher priority, update its priority and rebuild the heap.
            # If item already in priority queue with equal or lower priority, do nothing.
            # If item not in priority queue, do the same thing as self.push.
            for index, (p, c, i) in enumerate(self.heap):
                if item_id(i) == item_id(item):
                    if p <= priority:
                        break
                    del self.heap[index]
                    self.heap.append((priority, c, item))
                    heapq.heapify(self.heap)
                    break
            else:
                super().push(item, priority)

        
        def push(self, node):
            pos = node[0] # position of the node to the inserted
            if pos not in self.open_set and pos not in self.closed_set:
                super().push(node)
                self.open_set[pos] = node
            elif pos in self.closed_set:
                prev_state = self.closed_set[pos]
                if self.g_func(node) < self.g_func(prev_state):
                    del self.closed_set[pos]
                    self.open_set[pos] = node
                    super().push(node)
            else:
                # must be present in open set
                prev_state = self.open_set[pos]
                if self.f_func(node) < self.f_func(prev_state):
                    self.update(node, self.f_func(node), lambda x: x[0])

        def pop(self):
            node = super().pop()
            pos = node[0]
            del self.open_set[pos]
            self.closed_set[pos] = node
            return node
            

    return genericSearch(problem, aStarFringe(problem, heuristic), True)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
