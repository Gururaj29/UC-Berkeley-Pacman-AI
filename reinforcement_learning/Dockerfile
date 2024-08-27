FROM python:3.6.15

WORKDIR /usr/src/app

COPY . .

# CMD ["python3", "gridworld.py", "-g", "MazeGrid"]
#CMD ["python", "gridworld.py", "-a", "value", "-i", "100", "-k", "10"]
# CMD ["python", "gridworld.py", "-a", "value", "-i", "5"]
CMD ["python", "submission_autograder.py"]
# CMD ["python", "gridworld.py", "-a", "value", "-i", "100", "-g", "DiscountGrid", "--discount", "0.3", "--noise", "0.3", "-r", "0"]
# CMD ["python", "gridworld.py", "-a", "value", "-i", "100", "-g", "DiscountGrid", "--discount", "1", "--noise", "0.2", "-r", "-2"]

# CMD ["python", "gridworld.py", "-a", "asynchvalue", "-i", "1000", "-k", "10"]
# CMD ["python", "gridworld.py", "-a", "priosweepvalue", "-i", "1000"]
# CMD ["python", "gridworld.py", "-a", "q", "-k", "5", "-m"]
# CMD ["python", "crawler.py"]
# CMD ["python", "gridworld.py", "-a", "q", "-k", "100", "--noise", "0.0", "-e", "0.3"]

# CMD ["python", "gridworld.py", "-a", "q", "-k", "50", "-n", "0", "-g", "BridgeGrid", "-e", "0.5", "-l", "0.4"]
# CMD ["python", "pacman.py", "-p", "PacmanQAgent", "-x", "2000", "-n", "2010", "-l", "smallGrid"]
# CMD ["python", "pacman.py", "-p", "ApproximateQAgent", "-a", "extractor=SimpleExtractor", "-x", "50", "-n", "60", "-l", "mediumClassic"]