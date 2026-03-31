def convert_list_to_string(l):
    s = ''
    for e in l:
        s += ' ' + str(e)
    return s

class Solution:
    def equalPairs(self, grid: List[List[int]]) -> int:

        row_h_map = {}

        for row in grid:
            row_str = convert_list_to_string(row)
            if row_str in row_h_map:
                row_h_map[row_str] += 1
            else:
                row_h_map[row_str] = 1

        count = 0

        # print(row_h_map)

        for i in range(len(grid)):
            col = []

            for j in range(len(grid)):
                col.append(grid[j][i])

            col_str = convert_list_to_string(col)
            # print(col_str)

            if col_str in row_h_map:
                # print(col_str)
                count += row_h_map[col_str]

            # print('\n')

        return count
        