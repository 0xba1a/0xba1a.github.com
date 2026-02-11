class Solution:
    num_range = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    def reset(self):
        self.row_chk = {num: [] for num in range(9)}
        self.col_chk = {num: [] for num in range(9)}
        self.box_chk = {num: [] for num in range(9)}
        self.wrk_brd = None
        self.board = None


    def fill_chks(self):
        for row in range(9):
            for col in range(9):
                c = self.board[row][col]
                if c == '.':
                    continue

                self.row_chk[row].append(c)
                self.col_chk[col].append(c)
                box = col // 3
                box = box + 3 * (row // 3)
                self.box_chk[box].append(c)


    def get_possible_value(self, row, col, next_val='0'):
        box_index = (col // 3) + (3 * (row // 3))
        if next_val != '0':
            self.row_chk[row].remove(next_val)
            self.col_chk[col].remove(next_val)
            self.box_chk[box_index].remove(next_val)

        for num_char in self.num_range[self.num_range.index(next_val)+1:]:
            if num_char in self.row_chk[row] or num_char in self.col_chk[col] or num_char in self.box_chk[box_index]:
                continue
            self.row_chk[row].append(num_char)
            self.col_chk[col].append(num_char)
            self.box_chk[box_index].append(num_char)

            return num_char

        return None


    def solv_board(self, row, col):
        if col == 9 and row == 9:
            return True

        cur_brd = self.wrk_brd

        next_val = self.get_possible_value(row, col, '0')
        while next_val:
            cur_brd[row][col] = next_val
            n_row, n_col = self.get_next_empty_cell(row, col)
            if self.solv_board(n_row, n_col):
                return True
            cur_brd[row][col] = '.'
            next_val = self.get_possible_value(row, col, next_val)

        return False

    def get_next_empty_cell(self, row, col):

        while row < 9:
            while col < 9:
                if self.wrk_brd[row][col] == '.':
                    return row, col
                col += 1
            row += 1
            col = 0

        return 9, 9


    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        self.reset()

        self.board = board
        self.wrk_brd = board
        self.fill_chks()

        row, col = self.get_next_empty_cell(0,0)
        self.solv_board(row, col)
        return self.wrk_brd
