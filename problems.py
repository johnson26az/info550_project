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
helper functions
'''
# Normalizes the grid to ensure that blocked cells are represented by '#' and empty cells by ' '.
def normalize_grid(grid):
   return [
      [('#' if ch == '#' else ' ') for ch in row]
      for row in grid
   ]


'''
Represents the crossword puzzle.
'''
class Crossword:
   def __init__(self, grid, clues, word_bank):
      self.grid = grid
      self.clues = clues
      self.word_bank = word_bank
      self.rows = len(grid)
      self.cols = len(grid[0])

   def is_blocked(self, row, col):
      return self.grid[row][col] == '#'

   def get_clue(self, clue_id):
      return self.clues.get(clue_id)

   def set_cell_letter(self, row, col, value):
      if 0 <= row < self.rows and 0 <= col < self.cols:
         self.grid[row][col] = value

   def print_grid(self):
      print("\nCROSSWORD PUZZLE")
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

   def get_word_bank(self):
      return self.word_bank


class CrosswordPuzzle:
   def __init__(self, crossword):
      self.crossword = crossword
      self.slots = {}  # holds the crossword slots for each clue
      self.graph = {}  # this will hold the
      self.domain = {} # will hold the candidate words
      self.assignment = {} # will hold the current assignment of words to slots

      self.extract_slots()
      self.find_intersections()
      self.build_domain()

      '''
      # for debugging purposes
      print("\nDOMAIN CHECK:")
      for slot_id, domain in self.domain.items():
         if len(domain) == 0:
            print(f"EMPTY DOMAIN → {slot_id}")
         else:
            print(f"{slot_id}: {len(domain)} candidates")
      '''


   # extract all the slots from the crossword grid
   def extract_slots(self):
      # logic to extract slots from the crossword grid
      grid = self.crossword.grid
      rows = self.crossword.rows
      cols = self.crossword.cols

      # across slots
      for row in range(rows):
         for col in range(cols):
            is_start = (
               not self.crossword.is_blocked(row, col)
               and (col == 0 or self.crossword.is_blocked(row, col - 1))
            )
            if is_start:
               start = col
               end = col
               while end < cols and not self.crossword.is_blocked(row, end):
                  end += 1
               length = end - start

               if length > 1:
                  slot_id = f"across_{row+1}-{start+1}"
                  cells = [(row, c) for c in range(start, start + length)]
                  self.slots[slot_id] = {
                     "id": slot_id,
                     "direction": "across",
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
            if is_start:
               start = row
               end = row
               while end < rows and not self.crossword.is_blocked(end, col):
                  end += 1
               length = end - start

               if length > 1:
                  slot_id = f"down_{start+1}-{col+1}"
                  cells = [(r, col) for r in range(start, start + length)]
                  self.slots[slot_id] = {
                     "id": slot_id,
                     "direction": "down",
                     "cells": cells,
                     "length": length
                  }


   # find intersections between slots and build the graph structure.
   def find_intersections(self):
      # initializing the graph structure
      for slot_id in self.slots:
         self.graph[slot_id] = {
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
                  self.graph[s1]["neighbors"].append(s2)
                  self.graph[s2]["neighbors"].append(s1)

                  # record the intersection index pair
                  self.graph[s1]["intersections"][s2] = (index1, index2)
                  self.graph[s2]["intersections"][s1] = (index2, index1)


   # builds the domain for each slot based on the word bank and the crossword grid.
   def build_domain(self):
      word_bank = self.crossword.word_bank

      for slot_id, slot in self.slots.items():
         length = slot["length"]
         cells = slot["cells"]

         candidates = []
         for word in word_bank:
            if len(word) != length:
               continue

            fits = True

            for (row, col), letter in zip(cells, word):
               grid_letter = self.crossword.grid[row][col]

               # only reject if the grid has a FIXED letter that doesn't match the candidate word
               if grid_letter not in ['#', ' '] and grid_letter != letter:
                  fits = False
                  break

            if fits:
               candidates.append(word)

         self.domain[slot_id] = candidates


   def print_graph(self):
      print("\nCROSSWORD STRUCTURE\n")
      for slot_id, data in self.graph.items():
         print(f"Slot ID: {slot_id}")
         print(f"  Neighbors: {data['neighbors']}")
         print(f"  Intersections: {data['intersections']}")


   # important for building the crossword graph structure.
   # this is used by the Node2Vec agent to build the graph for embedding
   def build_graph(self):
      G = nx.Graph()

      # add nodes for each slot
      for slot_id, slot in self.slots.items():
         G.add_node(slot_id, type = "word", length = slot["length"])

      # add word nodes
      for word in self.crossword.word_bank:
         G.add_node(word, type = "word", length = len(word))

      # slot <=> word edges
      for slot_id, slot in self.slots.items():
         for word in self.crossword.word_bank:
            if len(word) == slot["length"]:
               G.add_edge(slot_id, word, weight = 1.0)

      # slot <=> slot edges (intersections)
      for s1, data in self.graph.items():
         for s2, (i1, i2) in data["intersections"].items():
            G.add_edge(s1, s2, weight = 2.0)

      # word <=> word edges (shared letters)
      words = self.crossword.word_bank

      for w1 in words:
         for w2 in words:
            if w1 != w2:
               shared = sum(1 for a, b in zip(w1, w2) if a == b)
               if shared > 0:
                  G.add_edge(w1, w2, weight = shared)

      self.G = G

   # check if the current assignment matches the correct solution
   def match_solution_grid(self, correct_solution):
      for r in range(self.crossword.rows):
         for c in range(self.crossword.cols):
            if self.crossword.grid[r][c] != correct_solution[r][c]:
               return False
      return True


class PuzzleGame:
   def __init__(self, puzzle, agent, max_attempts, correct_solution):
      self.problem = puzzle
      self.agent = agent
      self.max_attempts = max_attempts
      self.correct_solution = correct_solution
      self.initial_grid = [row[:] for row in puzzle.crossword.grid]

   def solve_the_puzzle(self):
      attempts = 0

      while attempts < self.max_attempts:
         self.problem.crossword.grid = [row[:] for row in self.initial_grid]
         self.problem.assignment = {}  # reset the assignment for each attempt
         agent = self.agent

         # the agent solving the crossword puzzle
         while len(self.problem.assignment) < len(self.problem.slots):
            move = agent.get_move(self.problem)
         
            if move is None:
               print("No more moves available.")
               break

            slot_id, word = move
            self.problem.assignment[slot_id] = word

         solution = self.problem.assignment
         agent.apply_solution_to_grid(solution)

         # checking the solution against the correct solution
         if self.problem.match_solution_grid(self.correct_solution):
            print("\nAGENT FOUND A CORRECT SOLUTION!")
            print(f"Solution after {attempts + 1} attempts:")
            self.problem.crossword.print_grid()
            break
         else:
            print(f"\nAttempt {attempts + 1}: AGENT DID NOT FIND A CORRECT SOLUTION.")
            attempts += 1
            print(f"current attempts: {attempts}")

      if attempts == self.max_attempts:
         print(agent.__str__())
         print("\nAGENT FAILED TO FIND A CORRECT SOLUTION AFTER MAX ATTEMPTS.")


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
      self.parent = None  # this will hold the parent requirement
      self.children = []


   def add_parent(self, parent):
      if parent and self.parent is None:
         self.parent = parent
         parent.add_child(self)  # ensure the parent also knows about this child


   def add_child(self, child):
      if child not in self.children:
         self.children.append(child)


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
      current = parent_id

      while current is not None:
         if current == child_id:
            return False

         req = self.get_requirement(current)

         if req is None or req.parent is None:
            break

         current = req.parent.req_id

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
            if other.req_level == "Subsystem"
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