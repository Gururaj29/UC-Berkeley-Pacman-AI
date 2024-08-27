# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        
        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        # current position
        currPos = currentGameState.getPacmanPosition()
        # new position
        newPos = successorGameState.getPacmanPosition()
        # extracting walls to get the size of the grid
        walls = currentGameState.getWalls()
        height, width = walls.height, walls.width
        maxDistance = height + width # maximum distance between newPos and closest food

        newGhostStates = successorGameState.getGhostStates()
        # list of positions of ghosts in the new state which are not scared
        newGhostPos = [i.getPosition() for i in newGhostStates if i.scaredTimer <= 0]
        
        # if there is non scared ghost in the new position, then avoid this position. Hence, the least evaluation value.
        if newPos in newGhostPos:
            return -1
        
        # if there is no ghost in the new position then 'Stop' action will only decrease the score, so avoid stopping
        if newPos == currPos:
            return 0
        
        # food positions in the current state
        foodPos = currentGameState.getFood().asList()
        # list of manhattan distances from food positions in the current state and the new position
        distancesFromFood = [manhattanDistance(newPos, f) for f in foodPos]
        # distance to the closest food, not including capsule here. 
        # Although eating capsule and then the scared ghost will increase the score,
        # new ghost could be respawned in any random position and can collide the pacman
        minFoodDistance = min(distancesFromFood) if distancesFromFood else 0

        # subtracting from max distance to give more eval score for the closest food, this value cannot be negative
        return maxDistance-minFoodDistance

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minimax(self, gameState, agentIndex, d):
        if gameState.isWin() or gameState.isLose() or d == self.depth:
            # no action taken in the terminal state
            return self.evaluationFunction(gameState), None
        
        n = gameState.getNumAgents()
        isPacman = agentIndex == 0
        # comparison function to get the best value based on whether the agent is pacman or not
        compFn = max if isPacman else min
        bestAction, bestValue = None, None
        # initial values for best value based on whether the agent is pacman or not
        bestValue = -10**9 if isPacman else 10**9
        for action in gameState.getLegalActions(agentIndex):
            nextState = gameState.generateSuccessor(agentIndex, action)
            # Child value and action. Child action is ignored. Increase depth if it's the last agent
            val, _ = self.minimax(nextState, (agentIndex + 1)%n, d + 1 if agentIndex == n-1 else d)
            # val is better than bestValue, then record bestValue and bestAction
            if compFn(val, bestValue) != bestValue:
                bestValue = compFn(val, bestValue)
                bestAction = action
        return bestValue, bestAction

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        _, bestAction = self.minimax(gameState, 0, 0)
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def alphaBeta(self, gameState, agentIndex, d, alpha, beta):
        if gameState.isWin() or gameState.isLose() or d == self.depth:
            # no action taken in the terminal state
            return self.evaluationFunction(gameState), None
        
        n = gameState.getNumAgents()
        isPacman = agentIndex == 0
        compFn = max if isPacman else min
        bestAction = None
        bestValue = -10**9 if isPacman else 10**9
        for action in gameState.getLegalActions(agentIndex):
            nextState = gameState.generateSuccessor(agentIndex, action)
            # Child value and action. Child action is ignored. Increase depth if it's the last agent
            val, _ = self.alphaBeta(nextState, (agentIndex + 1)%n, d + 1 if agentIndex == n-1 else d, alpha, beta)
            # val is better than bestValue, then record bestValue and bestAction
            if compFn(val, bestValue) != bestValue:
                bestValue = compFn(val, bestValue)
                bestAction = action
            # update alpha and beta based on whether it is a minimizer or maximizer
            alpha = max(alpha, bestValue) if isPacman else alpha
            beta = min(beta, bestValue) if not isPacman else beta

            # if beta < alpha, prune subsequent child nodes
            if beta < alpha:
                break
        return bestValue, bestAction


    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        _, bestAction = self.alphaBeta(gameState, 0, 0, -(10 ** 9), 10**9)
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # extract current position and size of the grid
    currPos = currentGameState.getPacmanPosition()
    gridSize = (currentGameState.getWalls().height, currentGameState.getWalls().width)
    maxDistance = gridSize[0] + gridSize[1]
    
    # get the closest food manhattan distance
    foodPos = currentGameState.getFood().asList()
    foodDistances = [manhattanDistance(currPos, i) for i in foodPos]
    closestFood = min(foodDistances) if foodDistances else 0

    # initial evalutation value, adds more weightage to the game state score
    # negative weight to the number of food and capsules to drive pacman towards the state where there is food or capsule
    evalValue = currentGameState.getScore() * 10 - (len(foodPos) + len(currentGameState.getCapsules())) * 2

    # considers ghost only if it's closer than ghostDistanceThreshold
    ghostDistanceThreshold = 3
    ghostPos = currentGameState.getGhostPositions()
    ghostDistances = [manhattanDistance(currPos, i) for i in ghostPos]
    closestGhost = min(ghostDistances) if ghostDistances else maxDistance
    if closestGhost < ghostDistanceThreshold:
        # if any ghost is closer than ghostDistanceThreshold, then only consider ghost and decrement value by 1
        evalValue -= 1
    else:
        # consider closest food distance only if there is no ghost around. 
        # Since state value should be inversely related to the state value, add (max distance - closest food distance). This is never negative
        evalValue += maxDistance - closestFood

    return evalValue

# Abbreviation
better = betterEvaluationFunction
