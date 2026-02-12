class Solution:
    char_seq = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    def update_checks(self):
        for i in range(9):
            for j in range(9):
                c = self.board[i][j]
                if c == '.': continue
                box_index = (j//3) + (3 * (i//3))
                self.row[i].add(c)
                self.col[j].add(c)
                self.box[box_index].add(c)

    def set_cell(self, i, j, val):
        self.board[i][j] = val
        box_index = (j//3) + (3 * (i//3))
        self.row[i].add(val)
        self.col[j].add(val)
        self.box[box_index].add(val)

    def clear_cell(self, i, j, val):
        self.board[i][j] = '.'
        box_index = (j//3) + (3 * (i//3))
        self.row[i].remove(val)
        self.col[j].remove(val)
        self.box[box_index].remove(val)

    def get_next_val(self, i, j, next_val):
        box_index = (j//3) + (3 * (i//3))
        index = int(next_val) + 1
        while index < 10:
            c = self.char_seq[index]
            if (
                c not in self.row[i]
                and c not in self.col[j]
                and c not in self.box[box_index]
            ):
                return c
            index += 1
        return None

    def get_next_empty_cell(self, i, j):
        while i < 9:
            while j < 9:
                if self.board[i][j] == '.':
                    return i, j
                j += 1
            i += 1
            j = 0
        return i, j

    def solv_sudoku(self, i, j):
        i, j = self.get_next_empty_cell(i, j)
        if i >= 9:
            # puzzle solved
            return True

        next_val = self.get_next_val(i, j, '0')
        while next_val:
            self.set_cell(i, j, next_val)
            if self.solv_sudoku(i, j):
                return True
            self.clear_cell(i, j, next_val)
            next_val = self.get_next_val(i, j, next_val)

        return False

    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """

        self.board = board

        self.row = { i:set() for i in range(9)}
        self.col = { i:set() for i in range(9)}
        self.box = { i:set() for i in range(9)}

        self.update_checks()

        self.solv_sudoku(0, 0)
        