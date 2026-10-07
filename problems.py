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
#
#######################################################################

'''
Represents the crossword puzzle.
'''
class Crossword:
   def __init__(self, grid, clues):
      self.grid = grid
      self.clues = clues
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
      for row in self.grid:
         print("".join(row))

   def print_clues(self):
      print("Clues:\n")
      print("Across:\n")
      for clue_id, clue in self.clues["across"].items():
         print(f"{clue_id}: {clue}")

      print("Down:\n")
      for clue_id, clue in self.clues["down"].items():
         print(f"{clue_id}: {clue}")


import csv
from pathlib import Path


'''
Defines the properties of a requirement
'''
class Requirement:
   def __init__(self, req_id, req_title, req_text):
      self.req_id = req_id
      self.req_title = req_title
      self.req_text = req_text

      # structure for the relationships
      self.parent = None

      # variables for CSP formalization
      self.variables = set()
      self.domain = {}
      self.constraints = []

   def add_req_text(self, req_text):
      self.req_text = req_text

   def add_parent(self, parent):
      self.parent.append(parent)

   def add_variable(self, variable, domain):
      self.variables.add(variable)
      self.domain[variable] = domain

   def add_constraint(self, variables, func):
      self.constraints.append((variables, func))


'''
Defines a set of requirements contained in a specification
'''
class Specification:
   def __init__(self):
      self.requirements = {}

   def add_requirement(self, req):
      self.requirements[req.req_id] = req

   def get_requirement(self, req_id):
      return self.requirements.get(req_id)

   '''
   Builds the CSP (Constraint Satisfaction Problem) from the requirements.
   '''
   def build_csp(self):
      variables = set()
      domain = {}
      constraints = []
      
      for req in self.requirements.values():
         variables |= req.variables
         domain.update(req.domain)
         constraints.extend(req.constraints)

      return {
         "variables": list(variables),
         "domain": domain,
         "constraints": constraints
      }

   '''
   Loads the requirements from a CSV file.
   '''
   def load_req(self, file_path):
    with open(file_path, newline="", encoding="cp1252") as csvfile:  # the csv file is written in cp1252 encoding
        reader = csv.DictReader(csvfile)

        for index, row in enumerate(reader, start=1):
            req = Requirement(
                req_id = index,
                req_title = row["Title"],
                req_text = row["Requirement Text"],
            )
            self.add_requirement(req)


def main():
   '''
   # getting the input file path
   input_file_name = 'requirements_only.csv'
   input_file = Path(__file__).parent / "data" / input_file_name

   # try to see if the input file exists
   if not input_file.exists():
      raise FileNotFoundError(f"Input file not found: {input_file}")

   # establishing the specification and loading the requirements
   spec = Specification()
   spec.load_req(input_file)

   # debug - check if the requirements were loaded correctly
   print(f"Loaded {len(spec.requirements)} requirements")

   csp = spec.build_csp()
   print(csp)
   '''

   from crosswords.puzzle01 import crossword # loading the crossword puzzle

   cw = Crossword(
      grid = [list(row) for row in crossword["grid"]], # loads the grid
      clues = crossword["clues"] # loads the clues
   )

   print("\nCROSSWORD GRID\n")
   cw.print_grid()
   cw.print_clues()
   


if __name__ == "__main__":
   main()