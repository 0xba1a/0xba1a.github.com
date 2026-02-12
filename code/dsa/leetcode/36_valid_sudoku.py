class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        row = {i: set() for i in range(9)}
        col = {i: set() for i in range(9)}
        box = {i: set() for i in range(9)}

        for i in range(9):
            for j in range(9):
                c = board[i][j]
                if c == '.': continue

                if (
                    c in row[i]
                    or c in col[j]
                    or c in box[(j // 3) + (3 * (i // 3))]
                ): return False

                row[i].add(c)
                col[j].add(c)
                box[(j // 3) + (3 * (i // 3))].add(c)

        return True
