def is_connected(w1, w2):
    diff = 0
    for i in range(len(w1)):
        if w1[i] != w2[i]:
            if diff > 0:
                return False
            diff += 1
    return True

def get_all_neighs(node, wordList):
    neighs = []
    for word in wordList:
        if is_connected(node, word):
            neighs.append(word)

    return neighs


def get_all_paths_via(node, parents, parent_level):
    if not parents[node]:
        return [[node]]

    alt_paths = []
    for parent in parents[node]:
        if parent_level[parent] >= parent_level[node]:
            continue

        for alt_path_to_parent in get_all_paths_via(parent, parents, parent_level):
            alt_paths.append(alt_path_to_parent + [node])

    return alt_paths


class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        if endWord not in wordList:
            return []

        parents = {node: set() for node in wordList}
        parents[beginWord] = set()
        parent_level = {node: float("inf") for node in wordList}
        parent_level[beginWord] = 0

        bfsq = deque()
        visited = []
        shortest_paths = []
        level = 0

        bfsq.append(beginWord)

        while bfsq:
            level += 1
            for _ in range(len(bfsq)):
                node = bfsq.popleft()
                visited.append(node)

                for neigh in get_all_neighs(node, wordList):
                    if neigh == endWord:
                        for alt_path in get_all_paths_via(node, parents, parent_level):
                            shortest_paths.append(alt_path + [neigh])

                    elif neigh in visited:
                        continue

                    if parent_level[neigh] < level:
                        continue

                    parent_level[neigh] = level
                    parents[neigh].add(node)
                    if neigh not in bfsq:
                        bfsq.append(neigh)

            if len(shortest_paths):
                return shortest_paths

        return shortest_paths