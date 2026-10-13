'''
This file will contain implementations of each algorithm/agent type.
'''
import random

class RandomAgent:
  def __str__(self):
    return "Random Agent"
  def getMove(self, problem):
    moves = problem.getLegalMoves()
    return moves[random.randrange(len(moves))]


#######################################################################
# Addition to the template for INFO 450/550.
# algorithms.py
#######################################################################

import numpy as np

from node2vec import Node2Vec

# compute cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
  cosine_similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
  
  return cosine_similarity

'''
resources for understanding and using the Node2Vec. 
  https://www.geeksforgeeks.org/machine-learning/node2vec-algorithm/
  https://spotintelligence.com/2024/01/18/node2vec/

this will be the agent solving the crossword puzzle. it uses the word bank to generate embeddings for each word and
then leverages these embeddings to make informed decisions about word placement.

refactor the node2vec agent to be simple and act like a greedy agent. it does not utilize backtracking or any other search strategy.
it simply selects the next unassigned slot and ranks the candidate words based on their embeddings.
it then selects the highest-ranked word that is consistent with the current assignment.
'''
class Node2VecAgent:
  def __init__(self, puzzle):
    self.puzzle = puzzle
    self.model = None


  def __str__(self):
    return "Node2Vec Agent"


  # abstract interface that returns a single move for the agent to make. this will be used by the problem to get the next move from the agent.
  def get_move(self, problem):
    # ensuring the graph exists
    if not hasattr(problem, "G"):
      problem.build_graph()

    # training embeddeings once for the puzzle graph
    if self.model is None:
      self._train_embedding()

    #  pick the next unassigned slot
    slot_id = self.select_unassigned_slot(problem.assignment)

    # rank the candidate words for the slot based on their embeddings
    for word, _ in self._rank_candidates(slot_id):
      # check if the word is consistent with the current assignment
      if self._is_consistent(slot_id, word, problem.assignment):
        return (slot_id, word)

    # no consistent found
    return None


  # train the Node2Vec embeddings
  def _train_embedding(self):  
    if not hasattr(self.puzzle, "G"):
      raise ValueError("The puzzle graph is not built.")

    node2vec = Node2Vec(
      self.puzzle.G,
      dimensions = 128, # embedding dimension
      walk_length = 80, # nodes perwalk trajectory
      num_walks = 10, # walks starting from each node
      p = 1.0, # return parameter
      q =  1.0, # in-out parameter
      workers = 4, # number of parallel workers
    )

    self.model = node2vec.fit(window=10, min_count=1)


  # score the similarity between two nodes based on their embeddings
  def _score_word(self, slot_id, word):
    slot_vec = self.model.wv[slot_id]
    word_vec = self.model.wv[word]
  
    return cosine_similarity(slot_vec, word_vec)


  # rank the candidate words for a slot
  def _rank_candidates(self, slot_id):
    candidates = self.puzzle.domain[slot_id]
    scored = [(word, self._score_word(slot_id, word)) for word in candidates]
    scored.sort(key = lambda x: x[1], reverse = True)
    
    return scored


  # checking for consistency for the already assigned words.
  # this will ensure that the word fits all the intersections with already assigned words.
  def _is_consistent(self, slot_id, word, assignment):
    # intesections are neighbor_slot_id -> (index_self, index_neighbor)
    for neighbor_id, (index_self, index_neighbor) in self.puzzle.graph[slot_id]["intersections"].items():
      if neighbor_id in assignment:
        if word[index_self] != assignment[neighbor_id][index_neighbor]:
          return False
    
    return True


  # simple slot selection strategy: select the first unassigned slot.
  def select_unassigned_slot(self, assignment):
    for slot_id in self.puzzle.slots:
      if slot_id not in assignment:
        return slot_id

    return None
  

  # this will apply the solution to the crossword grid for ease of readability
  def apply_solution_to_grid(self, solution):
    cw = self.puzzle.crossword

    for slot_id, word in solution.items():
      slot = self.puzzle.slots[slot_id]
      cells = slot["cells"]
      
      for (row, col), letter in zip(cells, word):
        cw.set_cell_letter(row, col, letter)