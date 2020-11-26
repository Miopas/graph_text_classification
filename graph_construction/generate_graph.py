import spacy
import pandas as pd
import json
import sys
import re
import string
import pickle
from str_utils import preprocess

# Load a UD-based english model
nlp = spacy.load("en_ud_model_lg") # here you can change it to md/sm as you preffer

# Add BART converter to spaCy's pipeline
from pybart.api import Converter
converter = Converter()
nlp.add_pipe(converter, name="BART")

def process_doc(doc, graph):
    # Test the new converter component
    parsed_objs = []
    for i in range(len(doc)):
        tok = str(doc[i])
        if tok == ' ' or len(tok) == 0:
            continue

        if tok not in graph:
            graph[tok] = {}

        # {'head': cookies, 'rel': 'case', 'src': 'UD', 'alt': None, 'unc': False}
        objs = {}
        for obj in doc[i]._.parent_list:
            #print(obj)
            head = str(obj['head'])
            rel = str(obj['rel'])

            if head == ' ' or len(head) == 0 or rel == ' ' or len(rel) == 0 or rel == 'punct':
                continue

            if head not in graph[tok]:
                graph[tok][head] = {}

            if rel not in graph[tok][head]:
                graph[tok][head][rel] = 0
            graph[tok][head][rel] += 1
            objs.append(dict(obj))

        parsed_objs.append(objs)
    return parsed_objs


if __name__ == '__main__':
    infile = sys.argv[1]

    try:
        df = pd.read_csv(infile)
    except:
        df = pd.read_csv(infile, lineterminator='\n')

    new_df = {'text':[], 'label':[]}
    g = {}
    preprocessed = {}
    for i, row in df.iterrows():
        text = preprocess(row['text'])
        label = row['label']
        if int(label) != 1:
            continue

        new_df['text'].append(text)
        new_df['label'].append(label)

        doc = nlp(text)
        parsed_objs = process_doc(doc, g)
        preprocessed[row['text']] = parsed_objs


    pd.DataFrame(new_df).to_csv(infile + '.preprocessed', index=False)

    # Write the graph to the output file
    fw = open(infile+'.pos.graph', 'w')
    for tok, parent_obj in g.items():
        for head, rel_obj in parent_obj.items():
            for rel, count in rel_obj.items():
                fw.write('{}\t{}\t{}\t{}\n'.format(tok, head, rel, count))

    fw.close()

    with open('{}.pickle'.format(infile), 'wb') as handle:
        pickle.dump(preprocessed, handle, protocol=pickle.HIGHEST_PROTOCOL)


