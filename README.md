# UC Berkeley Pac-Man AI Project

This repository contains the implementation of various Artificial Intelligence (AI) techniques for controlling Pac-Man agents in a simulated environment. Developed as part of UC Berkeley's CS188 Intro to AI course, [this project](http://ai.berkeley.edu/project_overview) explores fundamental AI concepts such as search algorithms, neural networks, and reinforcement learning.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Implemented Features](#implemented-features)
  - [Search Algorithms](#search-algorithms)
  - [Multi-Agent Pac-Man](#multi-agent-pac-man)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Neural Networks](#neural-networks)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)

## Introduction

The Pac-Man AI project is a comprehensive educational tool designed to teach fundamental AI concepts through the development of intelligent agents for the classic arcade game Pac-Man. This project involves implementing and experimenting with various AI techniques, including search algorithms, adversarial games, neural networks, and reinforcement learning, to create agents that can effectively navigate and make decisions in the Pac-Man environment.

## Project Structure

The project is divided into several parts, each focusing on a specific area of AI:
- **Search**: Implementing search algorithms like BFS, DFS, UCS, and A* to navigate Pac-Man through mazes.
- **Multi-Agent**: Developing strategies for Pac-Man and ghost agents using adversarial search techniques.
- **Reinforcement Learning**: Applying Q-learning and value iteration to train Pac-Man to maximize rewards over time.
- **Neural Networks**: Exploring neural network-based approaches to improve decision-making and strategy in the game.

## Implemented Features

### Search Algorithms
- **Depth-First Search (DFS)**: Explores the deepest nodes first.
- **Breadth-First Search (BFS)**: Explores all nodes at the present depth before moving deeper.
- **Uniform Cost Search (UCS)**: Expands the least-cost node first.
- **A* Search**: Combines UCS with heuristics for efficient pathfinding.

### Multi-Agent Pac-Man
- **Minimax Algorithm**: Implements adversarial search where Pac-Man plays against ghosts.
- **Expectimax Algorithm**: Models uncertainty by incorporating randomness in ghost behavior.

### Reinforcement Learning
- **Q-Learning**: An off-policy learning technique to train Pac-Man to find optimal policies.
- **Value Iteration**: Computes the best policy by iteratively improving the value function.

### Neural Networks
- **Deep Q-Networks (DQN)**: Uses neural networks to approximate Q-values for reinforcement learning, enabling Pac-Man to learn strategies directly from raw pixel input or other state representations.
- **Policy Gradient Methods**: Trains neural networks to directly optimize the policy that Pac-Man follows, allowing for more sophisticated decision-making strategies.

## Setup and Installation

To set up and run the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Gururaj29/UC-Berkeley-Pacman-AI.git
   ```
2. Navigate to the repository
```bash
   git clone https://github.com/Gururaj29/UC-Berkeley-Pacman-AI.git
   ```
3. Ensure you have Python 3.x installed on your system. Install any required dependencies.
4. Checkout to the desired modules and run the module using the autograder scripts.

### How to Run

To run the Pac-Man game with the implemented search AI algorithms, go to the search directory use the following command format:

```bash
   python autograder.py -p SearchAgent -a fn=bfs
   ```

Replace SearchAgent with the desired agent and bfs with the desired search function (e.g., dfs, ucs, astar).

For more details on available options, run:
```bash
   python pacman.py -h
   ```
