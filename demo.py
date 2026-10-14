import problems as prb
import algorithms as alg

'''
This file will contain the code needed to run a demonstration of your project. 

For the progress report, it should contain code that shows a brief demonstration of random agents attempting to solve each problem. 

For the final report, it should contain code that shows a brief demonstration of each algorithm attempting to solve each problem. 

In both cases, to make sure we can grade these all in time, please ensure that each demonstration is just long enough to show some
meaningful behavior. If your problem is chess playing, for example, you might show 5 moves from each player, where one player uses one 
algorithm and one player uses another. If your problem is snake, you might show game long enough for the player to eat two apples or die
(whichever comes first).
'''

#demoGame = prb.Game(prb.TicTacToe(),alg.RandomAgent(),alg.RandomAgent())
#demoGame.playGame()

#######################################################################
# Addition to the template for INFO 450/550.
# demo.py
#######################################################################

from crosswords.puzzle02 import crossword as crossword02
from crosswords.puzzle03 import crossword as crossword03

from problems import Crossword, CrosswordPuzzle, PuzzleGame, normalize_grid

from algorithms import Node2VecAgent, Node2VecBacktrackingWithForwardCheckingAgent

CORRECT_SOLUTION = [list(row) for row in crossword03["solution_grid"]]

cw = Crossword(
      grid = normalize_grid(crossword03["grid"]), # loads the grid
      clues = crossword03["clues"], # loads the clues
      word_bank = crossword03["word_bank"]
   )

cw.print_grid()
#cw.print_clues()

# making a graph representation of the crossword puzzle
puzzle = CrosswordPuzzle(cw)
puzzle.build_graph()
#puzzle.print_graph()

agent = Node2VecAgent(puzzle)

MAX_ATTEMPTS = 10

print(f"Running {agent.__str__()}...")
game = PuzzleGame(puzzle, agent, MAX_ATTEMPTS, CORRECT_SOLUTION)
game.solve_the_puzzle()

# running the Node2VecBacktrackingWithForwardCheckingAgent on the same puzzle
agent2 = Node2VecBacktrackingWithForwardCheckingAgent(puzzle)
print(f"Running {agent2.__str__()}...")
game2 = PuzzleGame(puzzle, agent2, MAX_ATTEMPTS, CORRECT_SOLUTION)
game2.solve_the_puzzle()

