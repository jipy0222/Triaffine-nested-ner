# calculate different features for attention patterns

if __name__ == '__main__':
    seq = 30

    # Only attend to inside-token
    print("insidetoken")
    insidetoken_dict = {}
    insidetokenlen_dict = {}
    for i in range(1, seq+1):
        for j in range(i, seq+1):
            if j-i+1 not in insidetoken_dict:
                insidetoken_dict[j-i+1] = [(i, j)]
                insidetokenlen_dict[j-i+1] = 1
            else:
                insidetoken_dict[j-i+1].append((i, j))
                insidetokenlen_dict[j-i+1] += 1

    print(insidetokenlen_dict)
    print("variance of attention:", len(insidetokenlen_dict))
    # for item in insidetoken_dict.keys():
    #     print(item, insidetoken_dict[item])

    total_span = 0
    total = 0
    for i in range(1, seq+1):
        for j in range(i, seq+1):
            total += j-i+1
            total_span += 1
    expectation = total/total_span
    print("expectation: ", expectation)
    print("attention pattern 'insidetoken': only cares about the length of span")

    # Only attend to sub-span
    print("subspan")

    def calsub(i, j):
        l = j-i+1
        t = 0
        for i in range(1, l+1):
            t += i
        return t

    subspan_dict = {}
    subspanlen_dict = {}
    for i in range(1, seq+1):
        for j in range(i, seq+1):
            num = calsub(i, j)
            if num not in subspan_dict:
                subspan_dict[num] = [(i, j)]
                subspanlen_dict[num] = 1
            else:
                subspan_dict[num].append((i, j))
                subspanlen_dict[num] += 1

    print(subspanlen_dict)
    print("variance of attention:", len(subspanlen_dict))
    # for item in subspan_dict.keys():
    #     print(item, subspan_dict[item])

    total_span = 0
    total = 0
    for i in range(1, seq+1):
        for j in range(i, seq+1):
            total += calsub(i, j)
            total_span += 1
    expectation = total/total_span
    print("expectation: ", expectation)
    print("attention pattern 'subspan': only cares about the length of span")

    # Only attend to super-span
    print("superspan")

    def calsuper(i, j):
        t = 0
        for l in range(1, i+1):
            for r in range(j, seq+1):
                t += 1
        return t

    superspan_dict = {}
    superspanlen_dict = {}
    for i in range(1, seq+1):
        for j in range(i, seq+1):
            num = calsuper(i, j)
            if num not in superspan_dict:
                superspan_dict[num] = [(i, j)]
                superspanlen_dict[num] = 1
            else:
                superspan_dict[num].append((i, j))
                superspanlen_dict[num] += 1

    print(superspanlen_dict)
    print("variance of attention:", len(superspanlen_dict))
    # for item in superspan_dict.keys():
    #     print(item, superspan_dict[item])

    total_span = 0
    total = 0
    for i in range(1, seq+1):
        for j in range(i, seq+1):
            total += calsuper(i, j)
            total_span += 1
    expectation = total/total_span
    print("expectation: ", expectation)
    print("attention pattern 'superspan': not only cares about the length of span, but also location")

    # Only attend to non-overlap span
    print("nonoverlapspan")

    def calnonoverlap(i, j):
        c1 = 0
        c2 = 0
        l = i-1
        for li in range(1, l+1):
            c1 += li
        r = seq-j
        for ri in range(1, r):
            c2 += ri
        t = c1+c2
        return t

    nonoverlapspan_dict = {}
    nonoverlapspanlen_dict = {}
    for i in range(1, seq+1):
        for j in range(i, seq+1):
            num = calnonoverlap(i, j)
            if num not in nonoverlapspan_dict:
                nonoverlapspan_dict[num] = [(i, j)]
                nonoverlapspanlen_dict[num] = 1
            else:
                nonoverlapspan_dict[num].append((i, j))
                nonoverlapspanlen_dict[num] += 1

    print(nonoverlapspanlen_dict)
    print("variance of attention:", len(nonoverlapspanlen_dict))
    # for item in nonoverlapspan_dict.keys():
    #     print(item, nonoverlapspan_dict[item])

    total_span = 0
    total = 0
    for i in range(1, seq+1):
        for j in range(i, seq+1):
            total += calnonoverlap(i, j)
            total_span += 1
    expectation = total/total_span
    print("expectation: ", expectation)
    print("attention pattern 'nonoverlapspan': not only cares about the length of span, but also location")

    # Only attend to siblings
    print("siblings")

    def calsiblings(i, j):
        t = i-1 + seq - j
        return t

    siblings_dict = {}
    siblingslen_dict = {}
    for i in range(1, seq+1):
        for j in range(i, seq+1):
            num = calsiblings(i, j)
            if num not in siblings_dict:
                siblings_dict[num] = [(i, j)]
                siblingslen_dict[num] = 1
            else:
                siblings_dict[num].append((i, j))
                siblingslen_dict[num] += 1

    print(siblingslen_dict)
    print("variance of attention:", len(siblingslen_dict))
    # for item in siblings_dict.keys():
    #     print(item, siblings_dict[item])

    total_span = 0
    total = 0
    for i in range(1, seq+1):
        for j in range(i, seq+1):
            total += calsiblings(i, j)
            total_span += 1
    expectation = total/total_span
    print("expectation: ", expectation)
    print("attention pattern 'siblings': only cares about the length of span")

    # Only attend to span with the same head or the same tail
    print("samehandt")

    def calsamehandt(i, j):
        t = seq - i + 1 + j - 1 + 1
        return t

    print(calsamehandt(10, 25))

    samehandt_dict = {}
    samehandtlen_dict = {}
    for i in range(1, seq+1):
        for j in range(i, seq+1):
            num = calsamehandt(i, j)
            if num not in samehandt_dict:
                samehandt_dict[num] = [(i, j)]
                samehandtlen_dict[num] = 1
            else:
                samehandt_dict[num].append((i, j))
                samehandtlen_dict[num] += 1

    print(samehandtlen_dict)
    print("variance of attention:", len(samehandtlen_dict))
    # for item in samehandt_dict.keys():
    #     print(item, samehandt_dict[item])

    total_span = 0
    total = 0
    for i in range(1, seq+1):
        for j in range(i, seq+1):
            total += calsamehandt(i, j)
            total_span += 1
    expectation = total/total_span
    print("expectation: ", expectation)
    print("attention pattern 'samehandt': only cares about the length of span")

    # Attend to tuple of spans (by different slicing) (refer to <htnn>)
    # Multi-head with different pattern
