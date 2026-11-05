import csv
import re
import time

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

from crosswords.puzzle01 import crossword as crossword01
from crosswords.puzzle02 import crossword as crossword02

from problems import Crossword, CrosswordPuzzle, PuzzleGame, normalize_grid

from algorithms import Node2VecAgent, Node2VecBacktrackingWithForwardCheckingAgent, Node2VecConflictDirectedBackjumpingWithNoGoodLearningAgent
from algorithms import Node2VecRequirementsDecompositionAgent
from algorithms import GreedyRandomAgent

from problems import Specification

from pathlib import Path

IS_RANDOM_AGENT = False # set to True to run the random agent else False to run the Node2Vec agent
SOLVE_CROSSWORD = False # set to True to run the crossword puzzle else False to run the requirements problem
INPUT_FILE_NAME = 'requirements_only.csv'
INPUT_FILE = Path(__file__).parent / "data" / INPUT_FILE_NAME
REQ_SOLUTION_NAME = 'requirements_solution.csv'
REQ_SOLUTION_FILE = Path(__file__).parent / "data" / REQ_SOLUTION_NAME

MAX_ATTEMPTS = 100

# supporting functions in the demo.py file
def print_elapsed_time(elapsed_seconds):
    print(f"Elapsed time: {elapsed_seconds:.2f} seconds")
    print(f"Elapsed time: {elapsed_seconds / 60:.2f} minutes\n")


print("\nStarting the demo...\n"\
      "Current set up is: \n"\
      f"IS_RANDOM_AGENT: {IS_RANDOM_AGENT}\n"\
      f"SOLVE_CROSSWORD: {SOLVE_CROSSWORD}\n")

if IS_RANDOM_AGENT:
    CORRECT_SOLUTION = [list(row) for row in crossword02["solution_grid"]]

    cw = Crossword(
        grid = normalize_grid(crossword02["grid"]), # loads the grid
        clues = crossword02["clues"], # loads the clues
        word_bank = crossword02["word_bank"]
    )

    cw.print_grid()

    puzzle = CrosswordPuzzle(cw)
    puzzle.build_graph()

    random_agent = GreedyRandomAgent()
    random_agent.set_puzzle(puzzle)
    print(f"Running {random_agent.__str__()}...")
    game = PuzzleGame(puzzle, random_agent, MAX_ATTEMPTS, CORRECT_SOLUTION)

    start_time = time.perf_counter()
    try:
        game.solve_the_puzzle()
    finally:
        elapsed_seconds = time.perf_counter() - start_time
        print_elapsed_time(elapsed_seconds)
    start_time = 0 # reseting the timer for the next problem


    # preping the requirements decomposition
    # try to see if the input file exists
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    spec = Specification()
    spec.load_req(INPUT_FILE)
    print(f"Loaded requirements from {INPUT_FILE}")
    spec.build_graph()

    random_agent.set_spec(spec)

    # loading the expected parents from the solution file
    if not REQ_SOLUTION_FILE.exists():
        raise FileNotFoundError(f"Solution file not found: {REQ_SOLUTION_FILE}")
    requirements_solution = spec.load_correct_solution(REQ_SOLUTION_FILE)

    start_time = time.perf_counter()
    solution = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        spec.reset_requirement_relationships()
        solution = random_agent.solve()
        is_correct, mismatches = spec.check_requirement_solution(
            solution, requirements_solution
        )

        if is_correct:
            print(
                f"\nAGENT FOUND A CORRECT REQUIREMENTS SOLUTION "
                f"after {attempt} attempt(s)."
            )
            break

        print(f"Attempt {attempt}: agent solution did not match requirements_solution.csv.")
    else:
        print(
            f"\nAGENT FAILED TO FIND A CORRECT REQUIREMENTS SOLUTION "
            f"after {MAX_ATTEMPTS} attempts."
        )
        if mismatches:
            print("Example mismatch:", mismatches[0])

    elapsed_seconds = time.perf_counter() - start_time
    print_elapsed_time(elapsed_seconds)

    print(f"{random_agent.__str__()} solution:")
    random_agent.print_solution(solution)


else:
    if SOLVE_CROSSWORD:
        CORRECT_SOLUTION = [list(row) for row in crossword02["solution_grid"]]
        MAX_ATTEMPTS = 10

        cw = Crossword(
            grid = normalize_grid(crossword02["grid"]), # loads the grid
            clues = crossword02["clues"], # loads the clues
            word_bank = crossword02["word_bank"]
        )

        cw.print_grid()
        #cw.print_clues()

        # making a graph representation of the crossword puzzle
        puzzle = CrosswordPuzzle(cw)
        puzzle.build_graph()
        #puzzle.print_graph()
        
        '''
        # for testing the Node2VecAgent on the crossword puzzle
        agent = Node2VecAgent(puzzle)

        print(f"Running {agent.__str__()}...")
        game = PuzzleGame(puzzle, agent, MAX_ATTEMPTS, CORRECT_SOLUTION)
        game.solve_the_puzzle()

        # adding timer for solving the puzzle
        start_time = time.perf_counter()
        try:
            game.solve_the_puzzle()
        finally:
            elapsed_seconds = time.perf_counter() - start_time
            print_elapsed_time(elapsed_seconds)
        '''

        # running the Node2VecBacktrackingWithForwardCheckingAgent on the same puzzle
        agent2 = Node2VecBacktrackingWithForwardCheckingAgent(puzzle)
        print(f"Running {agent2.__str__()}...")
        game2 = PuzzleGame(puzzle, agent2, MAX_ATTEMPTS, CORRECT_SOLUTION)

        # adding timer for solving the puzzle
        start_time = time.perf_counter()
        try:
            game2.solve_the_puzzle()
        finally:
            elapsed_seconds = time.perf_counter() - start_time
            print_elapsed_time(elapsed_seconds)

        start_time = 0 # reseting the timer for the next agent

        # running the Node2VecConflictDirectedBackjumpingWithNoGoodLearningAgent on the same puzzle
        agent3 = Node2VecConflictDirectedBackjumpingWithNoGoodLearningAgent(puzzle)
        print(f"Running {agent3.__str__()}...")
        game3 = PuzzleGame(puzzle, agent3, MAX_ATTEMPTS, CORRECT_SOLUTION)

        # adding timer for solving the puzzle
        start_time = time.perf_counter()
        try:
            game3.solve_the_puzzle()
        finally:
            elapsed_seconds = time.perf_counter() - start_time
            print_elapsed_time(elapsed_seconds)

    else:
        # for requirements problem
        # getting the input file path
        #input_file_name = 'requirements_only.csv'
        #input_file = Path(__file__).parent / "data" / input_file_name

        # try to see if the input file exists
        if not INPUT_FILE.exists():
            raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

        # establishing the specification and loading the requirements
        spec = Specification()
        spec.load_req(INPUT_FILE)
        # debug - check if the requirements were loaded correctly
        print(f"Loaded {len(spec.requirements)} requirements")

        # loading the expected parents from the solution file
        if not REQ_SOLUTION_FILE.exists():
            raise FileNotFoundError(f"Solution file not found: {REQ_SOLUTION_FILE}")
        requirements_solution = spec.load_correct_solution(REQ_SOLUTION_FILE)

        spec.build_graph()
        #spec.print_graph()

        # creating the agent
        agent_req_0 = Node2VecRequirementsDecompositionAgent(spec)
        print(f"Running {agent_req_0.__str__()}...")

        # adding timer for solving the requirement decomposition problem
        start_time = time.perf_counter()
        solution = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            spec.reset_requirement_relationships()
            solution = agent_req_0.solve()
            is_correct, mismatches = spec.check_requirement_solution(
                solution, requirements_solution
            )
        
            if is_correct:
                print(
                    f"\nAGENT FOUND A CORRECT REQUIREMENTS SOLUTION "
                    f"after {attempt} attempt(s)."
                )
                break
        
            print(f"Attempt {attempt}: agent solution did not match requirements_solution.csv.")
        else:
            print(
                f"\nAGENT FAILED TO FIND A CORRECT REQUIREMENTS SOLUTION "
                f"after {MAX_ATTEMPTS} attempts."
            )
            if mismatches:
                print("Example mismatch:", mismatches[0])
        
        elapsed_seconds = time.perf_counter() - start_time
        print_elapsed_time(elapsed_seconds)

        print(f"{agent_req_0.__str__()} solution:")
        agent_req_0.print_solution(solution)
