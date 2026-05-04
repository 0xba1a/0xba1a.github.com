def insert_to_stack(astroid, stack, top):
    if astroid < 0:
        ast_dir = "left"
    else:
        ast_dir = "right"

    while True:
        if not top:
            stack[top] = astroid
            return top + 1

        neigh = stack[top - 1]
        if neigh < 0:
            neigh_dir = "left"
        else:
            neigh_dir = "right"

        if (
            ast_dir == neigh_dir
            or (neigh_dir == "left" and ast_dir == "right")
        ):
            stack[top] = astroid
            return top + 1

        if abs(neigh) < abs(astroid):
            top -= 1
        elif abs(neigh) == abs(astroid):
            return top - 1
        else:
            # No change in the stack
            return top


class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        stack :list = list(asteroids)
        top = 0

        for astroid in asteroids:
            top = insert_to_stack(astroid, stack, top)

        return stack[:top]
        