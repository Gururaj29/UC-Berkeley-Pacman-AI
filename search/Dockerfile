FROM python:3.6.15

WORKDIR /usr/src/app

COPY . .

# CMD [ "python3", "./pacman.py", "-l", "tinyCorners", "-p", "SearchAgent", "-a", "fn=bfs,prob=CornersProblem"]

CMD [ "python3", "./eightpuzzle.py"]

# CMD [ "python3", "./pacman.py", "-l", "mediumCorners", "-p", "SearchAgent", "-a", "fn=bfs,prob=CornersProblem", "-z", ".5", "--frameTime", "0"]

# CMD [ "python3", "./pacman.py", "-l", "mediumCorners", "-p", "AStarCornersAgent", "-z", ".5"]

# CMD [ "python3", "./pacman.py", "-l", "trickySearch", "-p", "AStarFoodSearchAgent"]

# CMD [ "python3", "./pacman.py", "-l", "bigSearch", "-p", "ClosestDotSearchAgent", "-z", ".5"]