import sys

INPUT = sys.argv[1]
with open(INPUT) as f:
    for line in f:
        adj_list = [tok.split(',') for tok in line.split()]
        new_list = []
        num_tokens = len(adj_list) + 1
        for edge in adj_list:
            new_edge = edge.copy()
            new_edge[0] = new_edge[0][1:]
            new_edge[2] = 'left'
            rev_edge = []
            rev_edge.append(new_edge[1])
            rev_edge.append(new_edge[0])
            rev_edge.append('right')
            new_list.append(new_edge)
            new_list.append(rev_edge)
        for i in range(num_tokens):
            new_list.append([str(i),str(i),'self'])
        print(' '.join(['(' + ','.join(edge) + ')' for edge in new_list]))
