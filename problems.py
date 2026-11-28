'''
This file will contain your the code for your problems. 
'''

import numpy as np
import cv2

class Game:
    def __init__(self, problem, pZero, pOne,verbose=True):
      self.problem = problem      
      self.players = [pZero,pOne]   
      self.verbose = verbose   
      if self.verbose:
         self.problem.showState()               
    def playGame(self):
      pCur=0
      while self.problem.isTerminal()==False:           
         move = self.players[pCur].getMove(self.problem)           
         self.problem.doMove(move)
         if self.verbose:
            self.problem.showState()
            print(f"Move {self.problem.ticks}:")
            print(self.problem.state)
         pCur = np.abs(pCur-1)
        
      wIndex = self.problem.getWinner()
      winner = self.players[wIndex]
      if wIndex == -1:
         winner = "DRAW"
      #display final state
      print(f"The Winner is {winner} ({wIndex})!")
      if self.verbose:
         self.problem.showState(4000)
      return wIndex
              
class TicTacToe:
   def __init__(self):
      self.state = np.zeros((3,3))      
      self.ticks=-1      
   def getLegalMoves(self, state=None):
      if state is None:
         state = self.state
      moves = []
      mark = 1
      if np.sum(np.abs(state))%2!=0:
         mark = -1       
      for i in range(state.shape[0]):
         for j in range(state.shape[1]):
            if state[i,j]==0:
               moves.append((mark,(i,j)))
      return moves             
   def getSuccessor(self, move, state):
      mark = move[0]
      loc = move[1]
      state[loc] = mark
      return state
   def doMove(self,move):
      self.state = self.getSuccessor(move,self.state)
      self.ticks+=1
   def isTerminal(self, state=None):
      if state is None:
         state = self.state         
      terminal = False
      val = self.evalTerminal(state)      
      if val ==0 and np.sum(np.abs(state))==9:
         terminal = True
      if val!=0:
         terminal = True
      return terminal
   def evalTerminal(self, state=None):
      if state is None:
         state = self.state
      val = 0      
      
      pZeroWins = [np.max(np.sum(state,0))==3,np.max(np.sum(state,1))==3,np.trace(state)==3,np.trace(state[:,::-1])==3]
      pOneWins = [np.min(np.sum(state,0))==-3,np.min(np.sum(state,1))==-3,np.trace(state)==-3,np.trace(state[:,::-1])==-3]

      if np.any(pZeroWins):
         val = 1
      if np.any(pOneWins):
         val = -1
      return val
   def getWinner(self, state=None):
      if state is None:
         state = self.state
      val = self.evalTerminal(state)
      if val==1:
         return 0 
      elif val==-1:
         return 1
      else:
         return -1
   def showState(self, ms = 1000,state=None):
      if state is None:
         state = self.state  

      screen = np.zeros((150,150)).astype(np.uint8)         
      screen = cv2.line(screen,(0,49),(149,49),255) 
      screen = cv2.line(screen,(0,99),(149,99),255)
      screen = cv2.line(screen,(49,0),(49,149),255) 
      screen = cv2.line(screen,(99,0),(99,149),255)

      for i in range(state.shape[0]):
         for j in range(state.shape[1]):
            if state[i,j]==1:
               screen = cv2.line(screen,(5 +j*50,5+i*50),(44+j*50,44+i*50),255,2,cv2.LINE_AA) 
               screen = cv2.line(screen,(44+j*50,5+i*50),(5 +j*50,44+i*50),255,2,cv2.LINE_AA) 
            if state[i,j]==-1:
               screen = cv2.circle(screen, (25 +j*50,25+i*50), 20, 255, 2,cv2.LINE_AA) 
               
      cv2.imshow('TicTacToe',screen)
      cv2.waitKey(ms)


#######################################################################
# Addition to the template for INFO 450/550.
# problems.py
#######################################################################

'''
Represents the crossword puzzle.
'''
class Crossword:
   def __init__(self, grid, clues):
      self.grid = [list(row) for row in grid]
      self.clues = clues

      self.rows = len(grid)
      self.cols = len(grid[0])


   def is_blocked(self, row, col):
      return self.grid[row][col] == '#'


   def get_clue(self, variable):
      direction, clue_number = variable
      return self.clues[direction][clue_number]


   def set_cell_letter(self, row, col, value):
      if 0 <= row < self.rows and 0 <= col < self.cols:
         self.grid[row][col] = value


   def print_grid(self):
      print("\nCROSSWORD PUZZLE\n")
      for row in self.grid:
         print("".join(row))
      print("\n")


   def print_clues(self):
      print("Clues:\n")
      print("Across:\n")
      for clue_id, clue in self.clues["across"].items():
         print(f"{clue_id}: {clue}")

      print("Down:\n")
      for clue_id, clue in self.clues["down"].items():
         print(f"{clue_id}: {clue}")


'''
crossword puzzle class formalizes the crossword puzzle as a CSP with variables, domains, and constraints.
'''
class CrosswordPuzzle:
   def __init__(self, crossword):
      self.crossword = crossword

      self.numbering = {}  # holds the numbering of the slots in the crossword grid
      self.slots = {}  # holds the crossword slots for each clue
      self.constraint_graph = {}  # this will hold the neighbor slots, intersections, and indices
      
      self.number_grid()  # number the slots in the crossword grid
      self.extract_slots()
      self.find_intersections()

      # csp definitions for the problem
      self.variables = list(self.slots.keys())  # list of slot IDs for the CSP
      self.domains = {
         variable: set()  # will hold the possible words for each slot
         for variable in self.variables
      }

      # metadata for CSP, holds the neighbors for each variable in the constraint graph
      self.neighbors = {
         variable: set(self.constraint_graph[variable]["neighbors"])
         for variable in self.variables
      }

      # metadata for CSP, holds the length constraints for each variable
      self.length_constraints = {
         variable: self.slots[variable]["length"]
         for variable in self.variables
      }

      # metadata for CSP, holds the intersection indices for each variable
      self.intersections = {
         variable: self.constraint_graph[variable]["intersections"]
         for variable in self.variables
      }
      
      self.assignment = {} # will hold the current assignment of words to slots

   # extract all the slots from the crossword grid
   def extract_slots(self):
      # logic to extract slots from the crossword grid
      rows = self.crossword.rows
      cols = self.crossword.cols

      # across slots
      for row in range(rows):
         for col in range(cols):
            is_start = (
               not self.crossword.is_blocked(row, col)
               and (col == 0 or self.crossword.is_blocked(row, col - 1))
            )

            if not is_start:
               continue

            start = col
            end = col

            while (end < cols and not self.crossword.is_blocked(row, end)):
               end += 1
            
            length = end - start

            if length <= 1:
               continue

            clue_number = self.numbering[(row, start)]
            slot_id = ("across", clue_number)

            cells = [
               (row, c)
               for c in range(start, end)
            ]

            self.slots[slot_id] = {
               "id": slot_id,
               "direction": "across",
               "clue_number": clue_number,
               "cells": cells,
               "length": length
            }

      # down slots
      for col in range(cols):
         for row in range(rows):
            is_start = (
               not self.crossword.is_blocked(row, col)
               and (row == 0 or self.crossword.is_blocked(row - 1, col))
            )

            if not is_start:
               continue

            start = row
            end = row

            while (end < rows and not self.crossword.is_blocked(end, col)):
               end += 1
            
            length = end - start

            if length <= 1:
               continue

            clue_number = self.numbering[(start, col)]
            slot_id = ("down", clue_number)

            cells = [
               (r, col)
               for r in range(start, end)
            ]

            self.slots[slot_id] = {
               "id": slot_id,
               "direction": "down",
               "clue_number": clue_number,
               "cells": cells,
               "length": length
            }


   # get the slot information for a given variable (slot ID)
   def get_slot(self, variable):
      return self.slots.get(variable)


   # get the neighbors of a given variable (slot ID) in the constraint graph
   def get_neighbors(self, variable):
      return self.neighbors[variable]


   def get_intersections(self, variable1, variable2):
      return self.intersections[variable1][variable2]


   def get_domain(self, variable):
      return self.domains[variable]


   def set_domain(self, variable, domain):
      self.domains[variable] = list(domain)


   def get_length(self, variable):
      return self.length_constraints[variable]


   # number the slots in the crossword grid for clues.
   def number_grid(self):
      self.numbering = {}

      clue_number = 1

      for row in range(self.crossword.rows):
         for col in range(self.crossword.cols):
            if self.crossword.is_blocked(row, col):
               continue

            start_across = (
               col == 0 or self.crossword.is_blocked(row, col - 1)
            )
               
            start_down = (
               row == 0 or self.crossword.is_blocked(row - 1, col)
            )

            if start_across or start_down:
               self.numbering[(row, col)] = clue_number
               clue_number += 1


   # find intersections between slots and build the graph structure.
   def find_intersections(self):
      # initializing the graph structure
      self.constraint_graph = {}

      for slot_id in self.slots:
         self.constraint_graph[slot_id] = {
            "neighbors": [],
            "intersections": {} # this will be the intersections with other slots
         }

      # compare every pair of slots to find intersections
      slot_ids = list(self.slots.keys())

      for i in range(len(slot_ids)):
         for j in range(i + 1, len(slot_ids)):
            s1 = slot_ids[i]
            s2 = slot_ids[j]

            # check for intersections between the two slots
            cells1 = self.slots[s1]["cells"]
            cells2 = self.slots[s2]["cells"]

            # check each cell pair
            for index1, cell1 in enumerate(cells1):
               if cell1 in cells2:
                  index2 = cells2.index(cell1)

                  # record the neighbor relationship
                  self.constraint_graph[s1]["neighbors"].append(s2)
                  self.constraint_graph[s2]["neighbors"].append(s1)

                  # record the intersection index pair
                  self.constraint_graph[s1]["intersections"][s2] = (index1, index2)
                  self.constraint_graph[s2]["intersections"][s1] = (index2, index1)


   def validate_slots(self):
      clue_ids = {
         (direction, clue_number)
         for direction, clues in self.crossword.clues.items()
         for clue_number in clues
      }
      slot_ids = set(self.slots)
      clue_count = (
         len(self.crossword.clues["across"]) +
         len(self.crossword.clues["down"])
      )

      slot_count = len(slot_ids)

      print(f"Number of clues: {clue_count}")
      print(f"Number of slots: {slot_count}")

      clues_without_slots = clue_ids - slot_ids
      slots_without_clues = slot_ids - clue_ids
      if clues_without_slots:
         print(f"Clues without slots: {sorted(clues_without_slots)}")
      if slots_without_clues:
         print(f"Slots without clues: {sorted(slots_without_clues)}")

      return clue_ids == slot_ids


   # assign a value (word) to a variable (slot ID) in the current assignment
   def assign(self, variable, value):
      self.assignment[variable] = value


   # unassign a variable (slot ID) from the current assignment
   def unassign(self, variable):
      if variable in self.assignment:
         del self.assignment[variable]


   def is_assigned(self, variable):
      return variable in self.assignment
   

   # check if the current assignment matches the correct solution
   def match_solution_grid(self, correct_solution):
      for r in range(self.crossword.rows):
         for c in range(self.crossword.cols):
            if self.crossword.grid[r][c] != correct_solution[r][c]:
               return False
      return True


   # debug function to print the slots in the crossword puzzle
   def print_slots(self):
      print("\nCROSSWORD SLOTS\n")
      for slot_id, slot in self.slots.items():
         clue = self.crossword.get_clue(slot_id)
         print(f"Slot ID: {slot_id}")
         print(f"  Clue: {clue}")


   # debug function to print the constraint graph
   def print_constraint_graph(self):
      print("\nCONSTRAINT GRAPH\n")
      for slot_id, data in self.constraint_graph.items():
         print(f"Slot ID: {slot_id}")
         print(f"  Neighbors: {data['neighbors']}")
         print(f"  Intersections: {data['intersections']}\n")


   '''
   utility functions for debugging
   '''
   def extract_word_from_solution(self, variable, solution_grid):
      slot = self.slots[variable]
      letters = []

      for row, col in slot["cells"]:
         letters.append(solution_grid[row][col])

      return "".join(letters).upper()


   def get_domain_rank(self, variable, word):
      domain = self.domains[variable]

      for rank, (candidate, score) in enumerate(domain, start=1):
         if candidate == word:
            return rank

      return None


   def domain_contains_word(self, variable, word):
      domain_words = {
         candidate
         for candidate, score
         in self.get_domain(variable)
      }

      return word in domain_words


   def verify_domain_coverage(self, solution_grid):
      print("\nDOMAIN COVERAGE VERIFICATION\n")

      found = 0
      total = len(self.variables)
      missing_words = []

      for variable in self.variables:
         answer = self.extract_word_from_solution(variable, solution_grid)

         rank = self.get_domain_rank(variable, answer)

         if rank is not None:
            found += 1

            print(
               f"[FOUND] {variable}",
               f"-> {answer}",
               f" (rank: {rank})"
            )
         else:
            missing_words.append((variable, answer))

            print(
               f"[MISSING] {variable}",
               f"-> {answer}",
               "not found in domain"
            )

      coverage = (
         found / total * 100
         if total > 0 else 0
      )

      print("\nSummary:")
      print(f"Total variables: {total}")
      print(f"Found in domain: {found}")
      print(f"Coverage: {coverage:.2f}%")

      if missing_words:
         print("\nMissing words:")
         for variable, answer in missing_words:
            print(f"{variable} -> {answer}")

      return coverage


import networkx as nx
import csv
import re

'''
Defines the properties of a requirement
'''
class Requirement:
   def __init__(self, req_id, req_title, req_text, req_level=None):
      self.req_id = req_id
      self.req_title = req_title
      self.req_text = req_text
      self.req_level = req_level  # this will be set when building the decomposition tree

      # structure for the relationships
      self.parents = set()
      self.children = set()


   def add_parent(self, parent):
      if parent and parent not in self.parents:
         self.parents.add(parent)
         parent.add_child(self)  # ensure the parent also knows about this child


   def add_child(self, child):
      if child not in self.children:
         self.children.add(child)


   def infer_level(self):
      req_id = self.req_id.upper()

      if req_id.startswith("SY"):
         self.req_level = "System"
      elif req_id.startswith("FE"):
         self.req_level = "Subsystem"
      elif req_id.startswith("BE"):
         self.req_level = "Subsystem"
      elif req_id.startswith("SW"):
         self.req_level = "Component"
      elif req_id.startswith("NF"):
         self.req_level = "Non-Functional"

      return self.req_level


STOP_WORDS = {
   "a", "an", "the", "shall",  "must", "should", "and", "or", "but", "is",
   "will", "system", "be", "to", "of", "in", "on", "for", "with", "by", "as",
   "are", "by"
}

'''
Defines a set of requirements contained in a specification
'''
class Specification:
   def __init__(self):
      self.requirements = {}
      self.G = nx.Graph()  # graph structure for the requirements

      # csp structures
      self.variables = []
      self.domains = {}
      self.constraints = []


   def add_requirement(self, req):
      self.requirements[req.req_id] = req


   def get_requirement(self, req_id):
      return self.requirements.get(req_id)


   # loads the requirements from a CSV file.
   def load_req(self, file_path):
      with open(file_path, newline="", encoding="cp1252") as csvfile:  # the csv file is written in cp1252 encoding
         reader = csv.DictReader(csvfile)

         for row in reader:
            req = Requirement(
                req_id = row["Requirements ID"].strip(),
                req_title = row["Title"].strip(),
                req_text = row["Requirement Text"].strip(),
            )
            req.infer_level()

            self.add_requirement(req)


   def create_decomposition_variables(self):
      self.variables = [
         req.req_id
         for req in self.requirements.values()
         if req.req_level != "System"
      ]

      return self.variables


   # builds the domains for the CSP based on the parent candidates for each variable
   def build_domains(self):
      self.domains = {}

      for variable in self.variables:
         self.domains[variable] = (
            self.get_parent_candidates(variable)
         )

      return self.domains


   # constraint to ensure a requirement cannot be its own parent
   def constraint_no_self(self, child_id, parent_id):
      return child_id != parent_id


   # constraint to ensure the parent-child relationship respects the requirement levels
   def constraint_level(self, child_id, parent_id):
      child = self.get_requirement(child_id)
      parent = self.get_requirement(parent_id)

      if child is None or parent is None:
         return False

      if child.req_level == "Subsystem":
         return parent.req_level == "System"
      elif child.req_level == "Component":
         return parent.req_level == "Subsystem"
      elif child.req_level == "Non-Functional":
         return parent.req_level in [
            "System", "Subsystem", "Component"
         ]

      return False


   # constraint to ensure there are no cycles in the parent-child relationships
   def constraint_no_cycle(self, child_id, parent_id):
      stack = [parent_id]
      visited = set()

      while stack:
         current = stack.pop()

         if current == child_id:
            return False  # cycle detected

         if current in visited:
            continue

         visited.add(current)
         current_req = self.get_requirement(current)

         if current_req:
            stack.extend(parent.req_id for parent in current_req.parents)
      
      return True


   # building the constraints for the CSP
   def build_constraints(self):
      self.constraints = [
         self.constraint_no_self,
         self.constraint_level,
         self.constraint_no_cycle
      ]

      return self.constraints

   
   # builds the CSP (Constraint Satisfaction Problem) from the requirements.
   def build_csp(self):
      self.create_decomposition_variables()
      self.build_domains()
      self.build_constraints()
  
      return {
         "variables": self.variables,
         "domains": self.domains,
         "constraints": self.constraints
      }


   # checks if a given parent-child relationship satisfies all the constraints
   def satisfies_constraints(self, child_id, parent_id):
      for constraint in self.constraints:
         if not constraint(child_id, parent_id):
            return False

      return True


   def get_requirement_by_level(self, level):
      return [
         req
         for req in self.requirements.values()
         if req.req_level == level
      ]


   # Returns a list of candidate parent requirements for a given requirement.
   def get_parent_candidates(self, req_id):
      req = self.requirements.get(req_id)
      if not req:
         return []

      if req.req_level == "System":
         return [None]
      elif req.req_level == "Subsystem":
         return [
            other.req_id
            for other in self.requirements.values()
            if other.req_level == "System"
         ]
      elif req.req_level == "Component":
         return [
            other.req_id
            for other in self.requirements.values()
            if other.req_level in ["Subsystem", "Non-Functional"]
         ]
      elif req.req_level == "Non-Functional":
         return [
            other.req_id
            for other in self.requirements.values()
            if other.req_level in ["System", "Subsystem", "Component"]
         ]


   # tokenizes the text of a requirement into a set of unique words.
   def tokenize_text(self, text):
      words = re.findall(r'\b\w+\b', text.lower())
      return {
         word
         for word in words
         if (word not in STOP_WORDS and len(word) > 2)
      }


   # builds the graph structure for the requirements, which can be used for visualization or analysis
   def build_graph(self):
      self.G.clear()
      reqs = list(self.requirements.values())

      # for adding nodes to the graph
      for req in reqs:
         self.G.add_node(
            req.req_id,
            title=req.req_title,
            text = req.req_text,
            level = req.req_level
         )

      # for adding edges to the graph
      for i in range(len(reqs)):
         for j in range(i + 1, len(reqs)):
            req1 = reqs[i]
            req2 = reqs[j]

            # check if they share any words
            words1 = self.tokenize_text(req1.req_text)
            words2 = self.tokenize_text(req2.req_text)

            overlap = words1 & words2

            if overlap:
               self.G.add_edge(
                  req1.req_id,
                  req2.req_id,
                  weight=len(overlap)
               )


   # a function to get the legal moves for a given variable in the CSP
   def legal_values(self, variable):
      return [
         value
         for value in self.domains[variable]
         if self.satisfies_constraints(variable, value)
      ]


   # debug function to print the graph structure
   def print_graph(self):
      print("\nREQUIREMENTS GRAPH STRUCTURE\n")

      print("\nNodes:")
      for node in self.G.nodes():
         print(node)

      print("\nEdges:")
      for u, v, data in self.G.edges(data=True):
         print(f"{u} -- {v}, weight: {data['weight']}")


   # funcions for checking the solution against the correct solution
   # load correct solution from a CSV file
   def load_correct_solution(self, solution_file):
      expected_parents = {}

      with open(solution_file, newline="", encoding="utf-8-sig") as csvfile:
         reader = csv.DictReader(csvfile)

         if reader.fieldnames:
            reader.fieldnames = [
               field.lstrip("\ufeff")
               for field in reader.fieldnames
            ]

         for row in reader:
            requirement_id = row["Requirements ID"].strip()
            parent_text = row["Parent Requirement"].strip()

            if not parent_text:
               continue

            parent_ids = set()

            for parent in parent_text.split(","):
               parent = parent.strip()

               range_match = re.fullmatch(
                  r"([A-Za-z]+)(\d+)\s*[-–]\s*([A-Za-z]+)(\d+)",
                  parent
               )

               if range_match and range_match.group(1) == range_match.group(3):
                  prefix = range_match.group(1)
                  start = int(range_match.group(2))
                  end = int(range_match.group(4))

                  parent_ids.update(
                     f"{prefix}{number:02d}"
                     for number in range(start, end + 1)
                  )
               else:
                  parent_ids.add(parent)

            expected_parents[requirement_id] = parent_ids

      return expected_parents


   # return whether the agent solution matches the expected solution from the CSV file
   def check_requirement_solution(self, agent_solution, correct_solution):
      mismatches = []

      for requirement_id, allowed_parents in correct_solution.items():
         actual_parent = agent_solution.get(requirement_id)
         if isinstance(actual_parent, str):
            actual_parents = {actual_parent}
         else:
            actual_parents = set(actual_parent or ())

         if actual_parents != allowed_parents:
            mismatches.append(
               f"{requirement_id}: "
               f"expected parents "
               f"{sorted(allowed_parents)}, "
               f"got {sorted(actual_parents)}"
            )

      unexpected_requirements = set(agent_solution) - set(correct_solution)
      
      for requirement_id in sorted(unexpected_requirements):
         mismatches.append(
            f"{requirement_id}: "
            f"not present in CSV solution"
         )

      return (len(mismatches) == 0, mismatches)


   # clear requirement relationships
   def reset_requirement_relationships(self):
      for requirement in self.requirements.values():
         requirement.parents.clear()
         requirement.children.clear()