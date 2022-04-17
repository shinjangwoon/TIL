from collections import defaultdict


def solution(id_list, report, k):
    answer = []
    report = set(report)
    count = defaultdict(int)
    users = defaultdict(set)

    for r in report:
        a, b = r.split()
        if b not in users[a]:
            users[a].add(b)
            count[b] += 1
    for id in id_list:
        result = 0
        for j in users[id]:
            if count[j] >= k:
                result += 1
        answer.append(result)
    return answer
