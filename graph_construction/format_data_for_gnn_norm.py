import sys
import pandas as pd
import str_utils
import numpy as np
from get_embeddings import get_doc_embedding, get_word_embedding

def randim_initialization(data_dir):
    emb_size = 768

    # encode docs
    df = str_utils.read_csv('{}/doc.ids'.format(data_dir))
    n = len(df)
    embeddings = np.random.rand(n, emb_size)
    with open('{}/doc.nodes'.format(data_dir), 'w') as fw:
        for i, (doc_id, label) in enumerate(zip(df['doc_id'], df['label'])):
            fw.write('{}\t{}\t{}\n'.format(doc_id, '\t'.join([str(x) for x in embeddings[i].tolist()]), label))
        fw.close()

    df = str_utils.read_csv('{}/word.ids'.format(data_dir))
    n = len(df)
    embeddings = np.random.rand(n, emb_size)
    with open('{}/word.nodes'.format(data_dir), 'w') as fw:
        for i, word_id in enumerate(df['word_id']):
            fw.write('{}\t{}\n'.format(word_id, '\t'.join([str(x) for x in embeddings[i].tolist()])))
        fw.close()

def word2vec_initialization(data_dir, metamap_map):
    # encode docs
    df = str_utils.read_csv('{}/word.ids'.format(data_dir))
    with open('{}/word.nodes'.format(data_dir), 'w') as fw:
        for i, (word_id, word) in enumerate(zip(df['word_id'], df['word'])):
            word_embedding = get_word_embedding(word_id, word)
            fw.write('{}\t{}\n'.format(word_id, '\t'.join([str(x) for x in word_embedding])))
        fw.close()

    df = str_utils.read_csv('{}/doc.ids'.format(data_dir))
    with open('{}/doc.nodes'.format(data_dir), 'w') as fw:
        for i, (doc_id, text, label) in enumerate(zip(df['doc_id'], df['text'], df['label'])):
            doc_embedding = get_doc_embedding(text, metamap_map)
            fw.write('{}\t{}\t{}\n'.format(doc_id, '\t'.join([str(x) for x in doc_embedding]), label))
        fw.close()


if __name__ == '__main__':
    data_dir = sys.argv[1]

    #word2vec_initialization(data_dir, None)
