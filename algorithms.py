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


# backtracking algorithm per figure 6.5 in the textbook (page 215).
# will extend the node2vec agent later.
def backtracking_search(agent, assignment):
  if len(assignment) == len(agent.puzzle.slots):
    return assignment

  slot_id = agent.select_unassigned_variable(assignment)

  for word in agent.order_domain_values(slot_id):
    if agent.is_consistent(slot_id, word, assignment):
      assignment[slot_id] = word
      
      result = backtracking_search(agent, assignment)
      
      if result is not None:
        return result

      del assignment[slot_id]

  return None


# returns true if the no domain wipeout occurs for any
# unassigned variable after assigning the word to the slot.
def forward_checking(puzzle, slot_id, word, assignment, domains):
  intersections = puzzle.graph[slot_id]["intersections"]

  for neighbor_id, (index_self, index_neighbor) in intersections.items():
    if neighbor_id not in assignment:
      new_domain = []

      for candidate in domains[neighbor_id]:
        if candidate[index_neighbor] == word[index_self]:
          new_domain.append(candidate)

      domains[neighbor_id] = new_domain

      if len(new_domain) == 0:
        return False

  return True


# backtracking with forward checking
def backtracking_search_with_forward_checking(agent, assignment, domains):
  if agent.is_complete(assignment):
    return assignment

  slot_id = agent.select_unassigned_variable(assignment)

  for word in agent.order_domain_values(slot_id):
    if agent.is_consistent(slot_id, word, assignment):
      assignment[slot_id] = word
      saved_domains = {
        slot_id: list(values)
        for slot_id, values in domains.items()
      }

      if forward_checking(agent.puzzle, slot_id, word, assignment, domains):
        result = backtracking_search_with_forward_checking(agent, assignment, domains)
        
        if result is not None:
          return result

      del assignment[slot_id]

      domains.clear()
      domains.update(saved_domains)

  return None


'''
resources for understanding and using the Node2Vec. 
  https://www.geeksforgeeks.org/machine-learning/node2vec-algorithm/
  https://spotintelligence.com/2024/01/18/node2vec/

this will be the agent solving the crossword puzzle. it uses the word bank to generate embeddings for each word and
then leverages these embeddings to make informed decisions about word placement.

refactored the node2vec agent to be simple and act like a greedy agent. it does not utilize backtracking or any other search strategy.
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

    return None # no consistent found


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
  # this is the consistency check for the CSP problem.
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


# a node2vec agent with CSP structure.
# this will be extended to include for adding CSP algorithms.
class Node2VecCSPAgent(Node2VecAgent):
  def is_complete(self, assignment):
    return len(assignment) == len(self.puzzle.slots)

  def is_consistent(self, slot_id, word, assignment):
    return super()._is_consistent(slot_id, word, assignment)

  def order_domain_values(self, slot_id):
    return [word for word, _ in self._rank_candidates(slot_id)]

  def select_unassigned_variable(self, assignment):
    return super().select_unassigned_slot(assignment)

  # modified function for solving the specification decomposition problem. this will be used for the requirements decomposition problem.
  # this gives an MRV heuristics directly on requirements
  def select_unassigned_variable_req(self, assignment, domains):
    unassigned = [
      req_id
      for req_id in self.spec.requirements
      if req_id not in assignment
    ]

    return min(unassigned, key=lambda req_id: len(domains[req_id]))


# extention of the Node2VecAgent with backtracking search and forward checking to find a solution to the crossword puzzle.
class Node2VecBacktrackingWithForwardCheckingAgent(Node2VecCSPAgent):
  def __str__(self):
    return "Node2Vec Backtracking with Forward Checking Agent"

  def __init__(self, puzzle):
    super().__init__(puzzle)
    self.solution = None
  
  def solve(self):
    if not hasattr(self.puzzle, "G"):
      self.puzzle.build_graph()

    if self.model is None:
      self._train_embedding()

    assignment = {}
    domains = {slot_id: list(self.puzzle.domain[slot_id]) for slot_id in self.puzzle.slots}
    
    solution = backtracking_search_with_forward_checking(self, assignment, domains)
    
    if solution is not None:
      self.apply_solution_to_grid(solution)
    
    return solution

  # implementing the get_move method to return the next move based
  # on the solution found by the backtracking search with forward checking.
  # this should remain compatible to the original get_move method in the Node2VecAgent class.
  def get_move(self, problem):
    # finiding the solution once and storing it for future moves.
    if self.solution is None:
      print("Computing CSP solution using backtracking with forward checking...")
      self.solution = self.solve()

    # no solution found, return None
    if self.solution is None:
      return None

    # return the next unassigned slot
    for slot_id, word in self.solution.items():
      if slot_id not in problem.assignment:
        return (slot_id, word)

    return None


# a node2vec agent with conflict-directed backjumping and no-good learning.
class Node2VecConflictDirectedBackjumpingWithNoGoodLearningAgent(Node2VecCSPAgent):
  def __str__(self):
    return "Node2Vec Conflict-Directed Backjumping with No-Good Learning Agent"


  def __init__(self, puzzle):
    super().__init__(puzzle)
    self.solution = None
    self.score_cache = {}
    self.no_goods = set()  # Store no-goods as frozensets for immutability
    self.conflict_sets = { # store conflicts for each slot
      slot_id: set() for slot_id in self.puzzle.slots
    }
    # adding statistics for comparison and analysis
    self.no_good_hits = 0
    self.no_goods_learned = 0
    self.score_cache = {}


  def solve(self):
    # reseting the no-goods and conflict sets for a fresh run
    self.solution = None
    self.no_goods.clear()
    self.conflict_sets = {slot_id: set() for slot_id in self.puzzle.slots}
    self.no_good_hits = 0
    self.no_goods_learned = 0

    if not hasattr(self.puzzle, "G"):
      self.puzzle.build_graph()

    if self.model is None:
      self._train_embedding()

    assignment = {}
    domains = {slot_id: list(self.puzzle.domain[slot_id]) for slot_id in self.puzzle.slots}
    self.score_cache = {
      (slot_id, word): self._score_word(slot_id, word)
      for slot_id, candidates in domains.items()
      for word in candidates
    }
    
    solution, _ = self.conflict_directed_backjumping(assignment, domains)
    
    if solution is not None:
      self.apply_solution_to_grid(solution)
    
    return solution


  def get_move(self, problem):
    # Compute the complete CSP solution once, then expose it one assignment
    # at a time through the PuzzleGame interface.
    if self.solution is None:
      print("Computing CSP solution using conflict-directed backjumping with no-good learning...")
      self.solution = self.solve()

    if self.solution is None:
      return None

    for slot_id, word in self.solution.items():
      if slot_id not in problem.assignment:
        return slot_id, word

    return None


  # Check if the current assignment violates any no-good
  def violates_no_good(self, assignment):
    current = frozenset(assignment.items())

    for no_good in self.no_goods:
      if no_good.issubset(current):
        self.no_good_hits += 1
        return True

    return False


  # checks if the assignment is consistent with the current slot and word, considering the no-goods learned so far.
  def consistency_conflict(self, slot_id, word, assignment):
    conflicts = set()

    for neighbor_id, (index_self, index_neighbor) in self.puzzle.graph[slot_id]["intersections"].items():
      # if the neighbor is already assigned, check for consistency
      if neighbor_id not in assignment:
        continue

      # if the neighbor is assigned, check for consistency
      if word[index_self] != assignment[neighbor_id][index_neighbor]:
        conflicts.add(neighbor_id)

    return len(conflicts) == 0, conflicts


  def order_domain_values_cbj(self, slot_id, domains):
    candidates = domains[slot_id]

    scored = [
      (word, self.score_cache[(slot_id, word)])
      for word in candidates
    ]
    scored.sort(key=lambda item: item[1], reverse=True)

    return [word for word, _ in scored]


  def select_unassigned_variable(self, assignment):
    # Choose the slot with the fewest currently compatible candidates.
    # This preserves CBJ while avoiding large dead-end subtrees.
    unassigned = [
      slot_id
      for slot_id in self.puzzle.slots
      if slot_id not in assignment
    ]

    def remaining_values(slot_id):
      return sum(
        self.consistency_conflict(slot_id, word, assignment)[0]
        for word in self.puzzle.domain[slot_id]
      )

    return min(unassigned, key=remaining_values)


  # implementing conflict-directed backjumping with no-good learning
  # nogood learning is based on Constraint Porcessing by Rina Dechter, 2003.
  def conflict_directed_backjumping(self, assignment, domains):
    # if the assignment is complete, return it
    if self.is_complete(assignment):
      return dict(assignment), None

    # select the next unassigned variable
    slot_id = self.select_unassigned_variable(assignment)

    self.conflict_sets[slot_id] = set()

    # for each value in the domain of the selected variable,
    # check if it is consistent with the current assignment and does not violate any no-good.
    # this is important. the no-good learning will help to prune the search space and avoid exploring paths that are known to be inconsistent.
    for word in self.order_domain_values_cbj(slot_id, domains):
      consistent, culprits = self.consistency_conflict(slot_id, word, assignment)

      if not consistent:
        self.conflict_sets[slot_id].update(culprits)

        if any(c in assignment for c in culprits):

          learned_no_good = frozenset(
            [
              (var, assignment[var])
              for var in self.conflict_sets[slot_id]
              if var in assignment
            ]
            + [(slot_id, word)]
          )
          if learned_no_good and learned_no_good not in self.no_goods: # store the learned no-good only if it is not empty
            self.no_goods.add(learned_no_good)
            self.no_goods_learned += 1

        continue

      assignment[slot_id] = word

      # check for no-good violation
      if self.violates_no_good(assignment):
        assignment.pop(slot_id, None)
        continue

      result, conflict_set = self.conflict_directed_backjumping(assignment, domains)

      if result is not None:
        return result, None

      if conflict_set is not None:

        if slot_id not in conflict_set:
          assignment.pop(slot_id, None)

          return None, conflict_set

        conflict_set.discard(slot_id)

        self.conflict_sets[slot_id].update(conflict_set)

      assignment.pop(slot_id, None)

    # if we reach here, it means that we have explored all the values in the domain of the current slot and none of them led to a solution.
    # we will learn a new no-good based on the conflict set of the current slot and backjump to the most recent variable in the conflict set.
    learned_no_good = frozenset([
      (var, assignment[var])
      for var in self.conflict_sets[slot_id]
      if var in assignment
    ])

    if learned_no_good: # store the learned no-good only if it is not empty
      if learned_no_good not in self.no_goods:
        self.no_goods.add(learned_no_good)
        self.no_goods_learned += 1

    return None, set(self.conflict_sets[slot_id])




# node2vec agent for requirements problem
class Node2VecRequirementCSPAgent:
  def __str__(self):
    return "Node2Vec Requirement CSP Agent"

  def __init__(self, specification):
    self.spec = specification
    self.model = None
    self.csp = specification.build_csp()

  def _train_embedding(self):
    if self.spec.G.number_of_nodes() == 0:
      self.spec.build_graph()

    node2vec = Node2Vec(
      self.spec.G,
      dimensions = 128,
      walk_length = 80,
      num_walks = 10,
      p = 1.0,
      q = 1.0,
      workers = 4,
    )

    self.model = node2vec.fit(window=10, min_count=1)

  def is_complete(self, assignment):
    return len(assignment) == len(self.csp["variables"])

  def score_parent(self, req_id, parent_id):
    return self.model.wv.similarity(str(req_id), str(parent_id))

  def order_domain_values(self, req_id, domains):
    ranked = []

    for parent_id in domains[req_id]:
      if parent_id is None:
        ranked.append((None, -1))
        continue

      score = self.score_parent(req_id, parent_id)
      ranked.append((parent_id, score))

    ranked.sort(key=lambda x: x[1], reverse=True)
      
    return [parent_id for parent_id, _ in ranked]


  # selects the next unassigned variable based on MRV heuristic
  def select_unassigned_variable(self, assignment, domains):
    unassigned = [
      req_id
      for req_id in self.csp["variables"]
      if req_id not in assignment
    ]

    return min(unassigned, key=lambda req_id: len(domains[req_id]))


  # checks if adding the parent_id to the assignment for req_id would create a cycle in the assignment graph
  def creates_cycle(self, req_id, parent_id, assignment):
    current = parent_id

    while current is not None:
      if current == req_id:
        return True
      
      current = assignment.get(current)

    return False


  # checks if the assignment is consistent with the current req_id and parent_id, considering the no-goods learned so far.
  def is_consistent(self, req_id, parent_id, assignment):
    if parent_id is None:
      return True

    if req_id == parent_id:
      return False

    if self.creates_cycle(req_id, parent_id, assignment):
      return False

    child = self.spec.get_requirement(req_id)
    parent = self.spec.get_requirement(parent_id)

    if child.req_level == "Subsystem":
      if parent.req_level != "System":
        return False
      elif child.req_level == "Component":
        if parent.req_level not in ["System", "Subsystem"]:
          return False
      elif child.req_level == "Non-Functional":
        if parent.req_level not in ["System", "Subsystem", "Component"]:
          return False

    return True


  # applies the assignment of parent-child relationships to the specification requirements
  def apply_decomposition(self, assignment):
    # clear the existing parent-child relationships
    for req in self.spec.requirements.values():
      req.parent = None
      req.children = []

    # build new hierarchy based on the assignment
    for req_id, parent_id in assignment.items():

      if parent_id is None:
        continue

      child = self.spec.get_requirement(req_id)
      parent = self.spec.get_requirement(parent_id)

      child.add_parent(parent)


  def print_solution(self, assignment):
    print("\nRequirement Decomposition Solution:\n")

    for req_id, parent_id in assignment.items():
      req = self.spec.get_requirement(req_id)

      if parent_id is None:
        print(
          f"{req.req_id} "
          f"({req.req_level}) "
          f"-> ROOT"
        )
      else:
        parent = self.spec.get_requirement(parent_id)
        print(
          f"{req.req_id} "
          f"({req.req_level}) "
          f"-> "
          f"{parent.req_id} "
          f"({parent.req_level})"
        )


# agent for requirements decomposition problem. this will be used to solve the requirements decomposition problem using the node2vec embeddings
# this agent will use only the bactracking with forward checking functions
class Node2VecRequirementsDecompositionAgent(Node2VecRequirementCSPAgent):
  def __str__(self):
    return "Node2Vec Requirements Decomposition Agent using Backtracking with Forward Checking"

  def __init__(self, specification):
    super().__init__(specification)
    self.solution = None


  # forward checking for the requirements decomposition problem
  # since the consistency checker already identifies the cycles, we can 
  # prune any domain value that would create one
  def requirements_forward_checking(self, req_id, parent_id, assignment, domains):
    if parent_id is None:
      return True

    for variable in domains:
      if variable in assignment:
        continue

      new_domain = []

      for candidate_parent in domains[variable]:
        if candidate_parent is None:
          new_domain.append(candidate_parent)
          continue

        test_assignment = assignment.copy()
        test_assignment[variable] = (candidate_parent)

        if self.is_consistent(variable, candidate_parent, test_assignment):
          new_domain.append(candidate_parent)

      domains[variable] = new_domain

      if len(new_domain) == 0:
        return False
    
    return True


  def requirement_backtracking_with_forward_checking(self, assignment, domains):
    if self.is_complete(assignment):
      return assignment

    req_id = self.select_unassigned_variable(assignment, domains)

    for parent_id in self.order_domain_values(req_id, domains):
      if self.is_consistent(req_id, parent_id, assignment):
        assignment[req_id] = parent_id
        saved_domains = {
          var: list(values)
          for var, values in domains.items()
        }

        if self.requirements_forward_checking(req_id, parent_id, assignment, domains):
          result = self.requirement_backtracking_with_forward_checking(assignment, domains)

          if result is not None:
            return result

        del assignment[req_id]

        domains.clear()
        domains.update(saved_domains)

    return None


  # solves the requirements decomposition problem using backtracking with forward checking. this will return a valid assignment of parent-child relationships for the requirements.
  def solve(self):
    if self.model is None:
      self._train_embedding()

    assignment = {}

    domains = {
      var: list(values)
      for var, values in self.csp["domains"].items()
    }

    solution = self.requirement_backtracking_with_forward_checking(assignment, domains)
    self.solution = solution

    if solution:
      self.apply_decomposition(solution)

    return solution