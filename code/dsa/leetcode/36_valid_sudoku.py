class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        rows = {num: [] for num in range(1, 10)}
        cols = {num: [] for num in range(1, 10)}
        box = {num: [] for num in range(1, 10)}

        row = 1
        col = 1

        for line in board:
            col = 1
            for c in line:
                if c == '.':
                    col += 1
                    continue

                if c in rows[row]:
                    return False
                else:
                    rows[row].append(c)

                if c in cols[col]:
                    return False
                else:
                    cols[col].append(c)

                box_index = 1 + (col-1) // 3 # 3 -> 1, 6 -> 2, 9 -> 3
                box_index = box_index +  3 * ((row-1) // 3) # 3 -> 0, 6 -> 3, 9 -> 6
                if c in box[box_index]:
                    print(f"{c} at {row} and {col} is a repitition")
                    print(box)
                    print(box_index)
                    return False
                else:
                    box[box_index].append(c)

                col += 1
            row += 1

        return True

