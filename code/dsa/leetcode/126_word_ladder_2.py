class Solution:
    def is_connected(self, w1, w2):
        diff = 0

        for i in range(len(w1)):
            if w1[i] != w2[i]:
                if diff > 0:
                    return False
                diff += 1

        return True


    def get_all_neighbors(self, node, wordList):
        neighs = []
        for word in wordList:
            if word in self.al[node]:
                neighs.append(word)
            elif self.is_connected(node, word):
                self.al[node].add(word)
                self.al[word].add(node)
                neighs.append(word)

        return neighs


    def get_all_possible_paths(self, node, parents):
        if not parents[node]:
            return [[node]]

        alt_paths = []
        for parent in parents[node]:
            for alt_path_to_parent in self.get_all_possible_paths(parent, parents):
                alt_paths.append(alt_path_to_parent + [node])

        return alt_paths


    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        if endWord not in wordList:
            return []

        self.al = {node: set() for node in wordList}
        self.al[beginWord] = set()

        shortest_paths = []
        bfsq = deque()
        visited = []
        level = 0

        parents = {node: set() for node in wordList}
        parents[beginWord] = set()
        parent_levels = { node: float("inf") for node in wordList}
        parent_levels[beginWord] = 0

        bfsq.append(beginWord)

        while bfsq:
            level += 1
            for _ in range(len(bfsq)):
                node = bfsq.popleft()
                visited.append(node)

                for neigh in self.get_all_neighbors(node, wordList):
                    if neigh == endWord:
                        for alt_path in self.get_all_possible_paths(node, parents):
                            shortest_paths.append(alt_path + [endWord])

                    if neigh in visited:
                        continue

                    if parent_levels[neigh] < level:
                        continue

                    parent_levels[neigh] = level
                    parents[neigh].add(node)

                    if neigh not in bfsq:
                        bfsq.append(neigh)

            if len(shortest_paths) > 0:
                return shortest_paths

        return shortest_paths
