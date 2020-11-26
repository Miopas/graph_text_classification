import mmlrestclient
import sys
import pandas as pd
import argparse
import json
from str_utils import preprocess, read_csv

if __name__ == '__main__':
    infile = sys.argv[1]
    df = read_csv(infile)
    texts = df.text.tolist()
    texts = ['{}|{}'.format(i, preprocess(text.replace(r'\n', ''))) for i, text in enumerate(texts)]
    texts = '\n'.join(texts)
    args = mmlrestclient.construct_args(texts)
    response = mmlrestclient.process(args)
    #print(response.text)

    responses = response.text.split('\n')
    hashtbl = {}
    for r in responses:
        arr = r.split('|')
        if len(arr) != 10:
            continue
        raw_text = arr[6].split('-')[3].strip('"')
        norm_term = arr[3].lower()

        if raw_text not in hashtbl:
            hashtbl[raw_text] = {}
        if norm_term not in hashtbl[raw_text]:
            hashtbl[raw_text][norm_term] = 0
        hashtbl[raw_text][norm_term] += 1

    with open('{}.json'.format(infile), 'w') as fp:
        json.dump(hashtbl, fp)


