# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util
import math

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def getNewValueForState(self, state):
        """
            returns new value for a state using previous iteration values
        """
        valueFunc = lambda p, r, c: p * (r + self.discount * c)
        newValue = -math.inf
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            actionValue = 0
            for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                if self.mdp.isTerminal(nextState):
                    return self.mdp.getReward(state, action, nextState)
                actionValue += valueFunc(prob, self.mdp.getReward(state, action, nextState), self.getValue(nextState))
            newValue = max(newValue, actionValue)
            
        return newValue if newValue != -math.inf else 0
    
    def __runSingleIteration(self):
        """
            returns values after a single iteration
        """
        states = self.mdp.getStates()
        newValue = util.Counter()
        for state in states:
            newValue[state] = self.getNewValueForState(state)
        return newValue

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for iter in range(self.iterations):
            self.values = self.__runSingleIteration()

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qValueFunc = lambda p, r, v: p * (r + self.discount * v)
        q = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            q += (qValueFunc(prob, self.mdp.getReward(state, action, nextState), self.getValue(nextState)))
        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        if not actions:
            return None
        qValues = util.Counter()
        for action in actions:
            qValues[action] = self.getQValue(state, action)

        return qValues.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
    
    def __runSingleIteration(self, state):
        """
            updates value of the next state in each iteration
        """
        if not self.mdp.isTerminal(state):
            self.values[state] = self.getNewValueForState(state)
        return

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        numberOfStates, nextState = len(states), 0
        for iter in range(self.iterations):
            self.__runSingleIteration(states[nextState])
            nextState = (nextState+1)%numberOfStates
            



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        self.mdp = mdp
        self.states = self.mdp.getStates()
        self.maxQFunc = lambda state: max([self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)])
        self.predecessors = self.__getPredecessors()
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
    
    def __getPredecessors(self):
        predecessors = {}
        for state in self.states:
            for action in self.mdp.getPossibleActions(state):
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob != 0:
                        if nextState not in predecessors:
                            predecessors[nextState] = set()
                        predecessors[nextState].add(state)
        return predecessors
                    
    def __runSingleIteration(self, pQueue):
        state = pQueue.pop()
        self.values[state] = self.getNewValueForState(state)
        for p in self.predecessors[state]:
            diff = abs(self.getValue(p) - self.maxQFunc(p))
            if diff > self.theta:
                pQueue.update(p, -diff)


    def runValueIteration(self):
        priorityQueue = util.PriorityQueue()
        
        for state in self.states:
            if not self.mdp.isTerminal(state):
                diff = abs(self.getValue(state) - self.maxQFunc(state))
                priorityQueue.push(state, -diff)
        
        for iter in range(self.iterations):
            if priorityQueue.isEmpty():
                break
            self.__runSingleIteration(priorityQueue)
            

