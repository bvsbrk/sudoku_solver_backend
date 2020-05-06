"""
5
123

532


"""
"""
3  7  5  
 8 5   3 
7   9    
 4      8
52  4  1 
   6 8   
  9    42
      3  
       81
"""
from copy import deepcopy


class SudokuSolver:
    def __init__(self, sudoku):
        self.sudoku = sudoku
        self.can_change = [[True if self.sudoku[j][i] == ' ' else False for i in range(len(sudoku))] for j in
                           range(len(sudoku))]
        self.answers = []

    def validate(self):
        for x in self.sudoku:
            se = set()
            for y in x:
                if y in se:
                    return False
                if y != " ":
                    se.add(y)

        for i in range(len(self.sudoku)):
            se = set()
            for j in range(len(self.sudoku)):
                if self.sudoku[j][i] in se:
                    return False
                if self.sudoku[j][i] != " ":
                    se.add(self.sudoku[j][i])

        if len(self.sudoku) == 9:
            l = self.sudoku[0][0:3] + self.sudoku[1][0:3] + self.sudoku[2][0: 3]
            se = set()
            for x in l:
                if x != " ":
                    if x in se:
                        return False
                    se.add(x)
            l = self.sudoku[0][3:6] + self.sudoku[1][3:6] + self.sudoku[2][3: 6]
            se = set()
            for x in l:
                if x != " ":
                    if x in se:
                        return False
                    se.add(x)
            l = self.sudoku[0][6:9] + self.sudoku[1][6:9] + self.sudoku[2][6: 9]
            se = set()
            for x in l:
                if x != " ":
                    if x in se:
                        return False
                    se.add(x)

            l = self.sudoku[3][0:3] + self.sudoku[4][0:3] + self.sudoku[5][0: 3]
            se = set()
            for x in l:
                if x != " ":
                    if x in se:
                        return False
                    se.add(x)
            l = self.sudoku[3][3:6] + self.sudoku[4][3:6] + self.sudoku[5][3: 6]
            se = set()
            for x in l:
                if x != " ":
                    if x in se:
                        return False
                    se.add(x)
            l = self.sudoku[3][6:9] + self.sudoku[4][6:9] + self.sudoku[5][6: 9]
            se = set()
            for x in l:
                if x != " ":
                    if x in se:
                        return False
                    se.add(x)

            l = self.sudoku[6][0:3] + self.sudoku[7][0:3] + self.sudoku[8][0: 3]
            se = set()
            for x in l:
                if x != " ":
                    if x in se:
                        return False
                    se.add(x)
            l = self.sudoku[6][3:6] + self.sudoku[7][3:6] + self.sudoku[8][3: 6]
            se = set()
            for x in l:
                if x != " ":
                    if x in se:
                        return False
                    se.add(x)
            l = self.sudoku[6][6:9] + self.sudoku[7][6:9] + self.sudoku[8][6: 9]
            se = set()
            for x in l:
                if x != " ":
                    if x in se:
                        return False
                    se.add(x)

        return True

    def solve(self, idx):
        if idx >= len(self.sudoku) * len(self.sudoku):
            if self.validate():
                self.answers.append(deepcopy(self.sudoku))
        else:
            x, y = idx // len(self.sudoku), idx % len(self.sudoku)
            if self.can_change[x][y]:
                past = self.sudoku[x][y]
                for i in range(1, len(self.sudoku) + 1):
                    self.sudoku[x][y] = i
                    if self.validate():
                        self.solve(idx + 1)
                    self.sudoku[x][y] = past
            else:
                self.solve(idx + 1)

    def get_answer(self):
        self.solve(0)
        return self.answers
