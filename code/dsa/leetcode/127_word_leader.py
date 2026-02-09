def is_connected(w1, w2):
    diff = 0
    for i in range(len(w1)):
        if w1[i] != w2[i]:
            if diff > 0: return False
            diff += 1
    return True


class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if endWord not in wordList:
            return 0

        al = {node: set() for node in wordList}
        al[beginWord] = set()

        for word in wordList:
            for node in al:
                if node in al[word]:
                    continue
                if is_connected(node, word):
                    al[node].add(word)
                    al[word].add(node)

        distance = 0
        visited = []
        bfs_q = deque()
        bfs_q.append(beginWord)

        while bfs_q:
            # bfs_work_q = list(bfs_q)
            distance += 1
            # bfs_q = []
            for _ in range(len(bfs_q)):
            # for node in bfs_work_q:
                node = bfs_q.popleft()
                # if node in visited:
                #     continue
                visited.append(node)

                for neigh in al[node]:
                    if neigh == endWord:
                        return distance+1
                    if neigh in visited or neigh in bfs_q:
                        continue
                    bfs_q.append(neigh)

        return 0