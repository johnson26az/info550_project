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

# extending the RandomAgent to solve the crossoword and requirements
# decomposition problems using a greedy approach.
class GreedyRandomAgent(RandomAgent):
  def __str__(self):
    return "Greedy Random Agent"

  
  def __init__(self):
    self.puzzle = None
    self.spec = None
    self.model = None


  def set_puzzle(self, puzzle):
    self.puzzle = puzzle
    self.model = None


  def set_spec(self, spec):
    self.spec = spec
    self.model = None


  def initialize_embeddings(self, problem):
    if self.model is not None:
      return

    if not hasattr(problem, "build_graph"):
      return

    problem.build_graph()

    if problem.G.number_of_nodes() == 0:
      return

    if problem.G.number_of_edges() == 0:
      return

    node2vec = Node2Vec(
      problem.G,
      dimensions = 128,
      walk_length = 20,
      num_walks = 100,
      workers = 4
    )

    self.model = node2vec.fit(window=10, min_count=1)


  # this will check if the candidate word is consistent with the already assigned neighboring slots in the crossword puzzle
  def crossword_is_consistent(self, problem, slot_id, word):
    # check whether a candidate word agrees with already assigned neiboring slots
    for neighbor in problem.graph[slot_id]["neighbors"]:
      if neighbor not in problem.assignment:
        continue

      assigned_word = problem.assignment[neighbor]

      my_index, neighbor_index = problem.graph[slot_id]["intersections"][neighbor]

      if word[my_index] != assigned_word[neighbor_index]:
        return False

    return True


  # greedy heuristic score
  # for solving the crossword puzzle
  def crosssword_score_word(self, problem, slot_id, word):
    score = 0.0

    try:
      score += self.model.wv.similarity(slot_id, word)
    except KeyError:
      pass

    # reward compatibility with neighboring slots
    for neighbor in problem.graph[slot_id]["neighbors"]:
      if neighbor in problem.assignment:
        assigned_word = problem.assignment[neighbor]

        try:
          score += self.model.wv.similarity(word, assigned_word)
        except KeyError:
          pass
      else:
        my_index, neighbor_index = problem.graph[slot_id]["intersections"][neighbor]

        compatibility_count = 0

        for candidate in problem.domain[neighbor]:
          if candidate[neighbor_index] == word[my_index]:
            compatibility_count += 1

          score += compatibility_count

    return score


  def get_move(self, problem):

    # crossword puzzle
    if hasattr(problem, "slots"):
      return self._crossword_get_move(problem)
    
    # requirements decomposition problem
    elif hasattr(problem, "requirements"):
      return self._decomposition_get_move(problem)

    return None


  def _crossword_get_move(self, problem):
    if self.model is None:
      self.initialize_embeddings(problem)

    unassigned_slots = [
      slot_id
      for slot_id in problem.slots
      if slot_id not in problem.assignment
    ]

    if not unassigned_slots:
      return None

    # mrv heuristic
    slot_id = min(
      unassigned_slots,
      key=lambda s: len([
        w for w in problem.domain[s]
        if w not in problem.assignment.values()
      ])
    )

    # valid candidate words for the selected slot
    candidates = [
      word for word in problem.domain[slot_id]
      if word not in problem.assignment.values()
      and self.crossword_is_consistent(problem, slot_id, word)
    ]

    if not candidates:
      return None

    # score every candidate word based on the greedy heuristic
    scored_candidates = {}

    for word in candidates:
      scored_candidates[word] = self.crosssword_score_word(problem, slot_id, word)

    best_score = max(scored_candidates.values())
    best_words = [
      word for word, score in scored_candidates.items()
      if score == best_score
    ]

    # random tie-breaking among the best-scoring candidates
    chosen_word = random.choice(best_words)

    return slot_id, chosen_word
  

  # this will apply the solution to the crossword grid for ease of readability
  def apply_solution_to_grid(self, solution):
    cw = self.puzzle.crossword
  
    for slot_id, word in solution.items():
      slot = self.puzzle.slots[slot_id]
      cells = slot["cells"]
        
      for (row, col), letter in zip(cells, word):
        cw.set_cell_letter(row, col, letter)


  # functions for requirements decomposition problem
  def decomposition_score(self, problem, child_id, parent_id):
    child = problem.get_requirement(child_id)
    parent = problem.get_requirement(parent_id)

    child_words = problem.tokenize_text(child.req_text)
    parent_words = problem.tokenize_text(parent.req_text)

    overlap = len(child_words & parent_words)
    score = overlap * 2 # weighting the overlap more heavily

    if self.model is not None:
      try:
        score += max(
          0.0,
          self.model.wv.similarity(str(child_id), str(parent_id))
          )
      except KeyError:
        pass

    return score
  

  def _decomposition_get_move(self, problem):
    if self.model is None:
      self.initialize_embeddings(problem)

    unassigned = [
      variable
      for variable in problem.variables
      if len(problem.get_requirement(variable).parents) == 0
    ]

    if not unassigned:
      return None

    # mrv heuristic.  for optimization. will comment for now to make sure that we are purely using
    # greedy heuristic for parent selection.  this will be used to select the next child requirement to assign parents to.
    '''
    child_id = min(
      unassigned,
      key=lambda r:
        len(problem.legal_values(r))
    )
    '''

    child_id = random.choice(unassigned) # randomly selecting a child requirement to assign parents to.
    candidates = problem.legal_values(child_id)

    if not candidates:
      return None

    # greedy heuristic
    scores = {}
    for parent_id in candidates:
      scores[parent_id] = self.decomposition_score(problem, child_id, parent_id)

    # best value
    best_score = max(scores.values())

    # allow for some flexibility in parent selection, not just the absolute best
    # if the agent is assigning to many parents, this needs to be adjust to higher value
    threshold = best_score * 0.9

    chosen_parents = [
      parent_id
      for parent_id, score in scores.items()
      if score >= threshold
    ]

    return child_id, chosen_parents


  # this will apply the decomposition move to the specification requirements
  def apply_decomposition_move(self, problem, child_id, parent_ids):
    child = problem.get_requirement(child_id)

    for parent_id in parent_ids:
      parent = problem.get_requirement(parent_id)

      if child and parent:
        child.add_parent(parent)


  def solve(self):
    if self.spec is None:
      raise ValueError("A specification must be assigned before solving.")

    # building the csp
    self.spec.build_csp()

    assignment = {}

    while True:
      move = self._decomposition_get_move(self.spec)

      if move is None:
        break

      child_id, parent_ids = move

      self.apply_decomposition_move(self.spec, child_id, parent_ids)

      assignment[child_id] = set(parent_ids)

    return assignment


  def print_solution(self, assignment):
      print("\nRequirement Decomposition Solution:\n")
  
      for req_id, parent_ids in assignment.items():
        req = self.spec.get_requirement(req_id)
  
        if parent_ids is None:
          print(
            f"{req.req_id} "
            f"({req.req_level}) "
            f"-> ROOT"
          )
        else:
          parent_list = sorted(parent_ids)

          print(
            f"{req.req_id} ->"
            f" {', '.join(parent_list)}"
          )


# Node2Vec implementation
from node2vec import Node2Vec
import numpy as np

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

    return min(
      unassigned,
      key=lambda req_id: (
        len(domains[req_id]),
        -len(self.spec.G[req_id]) # tie-breaker: prefer requirements with more children
      )
    )


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


  # this will score the similarity between a child requirement and a parent requirement based on their embeddings
  def score_parent(self, req_id, parent_id):
    try:
      return self.model.wv.similarity(str(req_id), str(parent_id))
    except KeyError:
      return 0.0


  # orders the domain values (parent requirements) for a given child requirement based on their similarity scores
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


  # checks if adding the parent_id to the assignment for req_id would create a cycle in the assignment graph
  def creates_cycle(self, req_id, parent_id, assignment):
    pending = [parent_id]
    visited = set()

    while pending:
      current = pending.pop()

      if current is None or current in visited:
        continue

      if current == req_id:
        return True

      visited.add(current)
      assigned_parents = assignment.get(current, set())
      if isinstance(assigned_parents, str):
        assigned_parents = {assigned_parents}

      pending.extend(assigned_parents)

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
      if parent.req_level not in ["Subsystem", "Non-Functional"]:
        return False
    elif child.req_level == "Non-Functional":
      if parent.req_level not in ["System", "Subsystem", "Component"]:
        return False

    return True


  # applies the assignment of parent-child relationships to the specification requirements
  def apply_decomposition(self, assignment):
    # clear any existing parent-child relationships in the requirements before applying the new assignment
    for req in self.spec.requirements.values():
        req.parents.clear()
        req.children.clear()

    # allows the new decomposition
    for req_id, parent_ids in assignment.items():
        if parent_ids is None:
            continue

        if isinstance(parent_ids, str):
          parent_ids = {parent_ids}

        child = self.spec.get_requirement(req_id)
        if child is None:
          continue

        for parent_id in parent_ids:
          parent = self.spec.get_requirement(parent_id)
          if parent is not None:
            child.add_parent(parent)


  def print_solution(self, assignment):
    print("\nRequirement Decomposition Solution:\n")

    for req_id, parent_ids in assignment.items():
      req = self.spec.get_requirement(req_id)

      if not parent_ids:
        print(
          f"{req.req_id} -> ROOT"
        )
      else:
        if isinstance(parent_ids, str):
          parent_ids = {parent_ids}

        parents = [
          self.spec.get_requirement(parent_id)
          for parent_id in sorted(parent_ids)
        ]
        parent_text = ", ".join(
          parent.req_id
          for parent in parents
          if parent is not None
        )
        print(f"{req.req_id} -> {parent_text}")


# agent for requirements decomposition problem. this will be used to solve the requirements decomposition problem using the node2vec embeddings
# this agent will use only the bactracking with forward checking functions
class Node2VecRequirementsBacktrackingWithForwardCheckingAgent(Node2VecRequirementCSPAgent):
  def __str__(self):
    return "Node2Vec Requirements Decomposition Agent using Backtracking with Forward Checking"

  def __init__(self, specification):
    super().__init__(specification)
    self.solution = None


  # setting the child parent size limit
  def constraint_parent_count(self, parent_ids):
    return len(parent_ids) <= 13

  # this will select the next unassigned requirement to assign parents to.
  def select_unassigned_variable(self, assignment, domains):
    unassigned = []

    for req_id in self.csp["variables"]:
      if req_id not in assignment:
        unassigned.append(req_id)

    return unassigned[0] if unassigned else None


  # forward checking for the requirements decomposition problem
  # since the consistency checker already identifies the cycles, we can 
  # prune any domain value that would create one
  def requirements_forward_checking(self, assignment, domains):
    for variable in self.csp["variables"]:
      if variable in assignment:
        continue

      filtered = []

      for candidate_parent in domains[variable]:
        if self.is_consistent(variable, candidate_parent, assignment):
          filtered.append(candidate_parent)

      domains[variable] = filtered

      if len(filtered) == 0:
        return False

    return True


  # this will order the parent sets for a given requirement based on their similarity scores
  # the requirement solution has a child with upto 13 parents
  # the agent is strugling to find the correct parent sets for each requirement because it has to consider many possible combinations of parents
  # its making large search spaces for the agent to explore
  def order_parent_sets(self, req_id, domains):
    ranked = []

    for parent_id in domains[req_id]:
      if parent_id is None:
        continue

      score = self.score_parent(req_id, parent_id)
      ranked.append((parent_id, score))

    ranked.sort(key=lambda item: item[1], reverse=True)

    # ranks the parents based on their scores and returns a list of sets of parent_ids,
    # starting with the highest-scoring parent, then the top two, then the top three, etc.
    if not ranked:
      return []

    ranked_parents = []
    for parent_id, score in ranked:
      if parent_id is not None:
        ranked_parents.append(parent_id)

    ordered_parent_sets = []
    current_set = set()

    for parent_id in ranked_parents:
      current_set.add(parent_id)
      ordered_parent_sets.append(set(current_set))

    # try to return the largest parent set first
    #ordered_parent_sets.reverse()

    # sort the parents based on the size of the parent set and the total similarity score
    ordered_parent_sets.sort(
      key=lambda parent_set: (
        len(parent_set),
        sum(self.score_parent(req_id, parent_id) for parent_id in parent_set)
      ),
      reverse=True
    )

    return ordered_parent_sets

  
  def requirement_backtracking_with_forward_checking(self, assignment, domains):
    if self.is_complete(assignment):
      return assignment

    req_id = self.select_unassigned_variable(assignment, domains)

    for parent_ids in self.order_parent_sets(req_id, domains):

      # applying the constraint on the number of parents
      if not self.constraint_parent_count(parent_ids):
        continue

      if all(
        self.is_consistent(req_id, parent_id, assignment)
        for parent_id in parent_ids
      ):

        # for debugging. prints the current requirement and the parent set being tested
        #print(f"Testing {req_id} with parent set {sorted(parent_ids)}")
        
        assignment[req_id] = set(parent_ids)

        saved_domains = {
          var: list(values)
          for var, values in domains.items()
        }

        if self.requirements_forward_checking(assignment, domains):
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


# agent for requirements using only conflict-directed backjumping with no-good learning
class Node2VecRequirementsConflictDirectedBackjumpingWithNoGoodLearningAgent(Node2VecRequirementCSPAgent):
  def __str__(self):
    return "Node2Vec Requirements Decomposition Agent using Conflict-Directed Backjumping with No-Good Learning"

  def __init__(self, specification):
    super().__init__(specification)
    self.solution = None
    self.no_goods = set()
    self.conflict_sets = {
      req_id: set() for req_id in self.spec.requirements
    }
    self.no_good_hits = 0
    self.no_goods_learned = 0

  # need to work on this