import sys
import json
import str_utils


def get_norm(metamap_data, raw_text):
    norm = raw_text
    if raw_text in metamap_data:
        norms = metamap_data[raw_text]
        sorted_norms = sorted(norms.items(), key=lambda x:x[1], reverse=True)
        norm =  sorted_norms[0][0]
    norm = str_utils.process_digit(norm)
    return norm

if __name__ == '__main__':
    infile = sys.argv[1]
    metamap_dict = sys.argv[2]
    outfile = '{}.norm'.format(infile)

    # Load the metamap dict
    with open(metamap_dict, 'r') as fr:
        metamap_data = json.load(fr)

    # Normalize node texts and create a new graph
    graph = {}
    with open(infile) as fr:
        for line in fr:
            tok, head, rel, count = line.strip().split('\t')
            tok = get_norm(metamap_data, tok)
            head = get_norm(metamap_data, head)

            # Exclude 'root'
            if rel == 'root':
                continue

            if tok not in graph:
                graph[tok] = {}

            if head not in graph[tok]:
                graph[tok][head] = {}

            if rel not in graph[tok][head]:
                graph[tok][head][rel] = 0

            graph[tok][head][rel] += int(count)


    # Write the graph to the output file
    fw = open(outfile, 'w')
    for tok, parent_obj in graph.items():
        for head, rel_obj in parent_obj.items():
            # Deal with multiple links between two nodes
            max_rel = None
            max_count = 0
            total_count = 0
            for rel, count in rel_obj.items():
                if count > max_count:
                    max_count = count
                    max_rel = rel
                total_count += count
            fw.write('{}\t{}\t{}\t{}\n'.format(tok, head, max_rel, total_count))
    fw.close()

