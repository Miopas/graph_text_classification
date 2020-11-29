import sys
import pandas as pd
from str_utils import read_csv, preprocess
import math
import json
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import find
from sklearn.feature_extraction import text

from normalize_graph import get_norm
#from format_data_for_gnn import word2vec_initialization
from get_embeddings_norm import get_doc_embedding, get_word_embedding


def normalize_wc_map(wc_map, metamap_map):
    new = {}
    for word, count in wc_map.items():
        if word not in metamap_map:
            continue
        norm = metamap_map[word]
        if norm not in new:
            new[norm] = count
        else:
            new[norm] += count
    return new

def metamap_normalize(metamap_dict, vocab):
    new = []
    metamap_map = {}
    with open(metamap_dict, 'r') as fr:
        metamap_data = json.load(fr)
    for word in vocab:
        norm = get_norm(metamap_data, word)
        new.append(norm)
        metamap_map[word] = norm
    new = list(set(new))
    return new, metamap_map

def get_tfidf(docs):
    #vectorizer = TfidfVectorizer(stop_words=text.ENGLISH_STOP_WORDS)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(docs)
    return X, vectorizer.get_feature_names(), vectorizer

def get_count(docs):
    #vectorizer = CountVectorizer(stop_words=text.ENGLISH_STOP_WORDS)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)
    vocab = vectorizer.get_feature_names()
    wc_map = {}
    X = X.toarray()
    for i, w in enumerate(vocab):
        count = sum(X[:,i])
        wc_map[w] = count
    return wc_map, vocab

def get_pmi(pair_count, a_count, b_count, total_count):
    p_i = a_count/total_count
    p_j = b_count/total_count
    p_ij = pair_count/total_count
    pmi = math.log(p_ij/(p_i*p_j))
    return pmi

if __name__ == '__main__':
    data_path = sys.argv[1]

    train = '{}/train.csv'.format(data_path)
    dev = '{}/dev.csv'.format(data_path)
    test = '{}/test.csv'.format(data_path)
    metamap_dict = '{}/train.csv.json'.format(data_path)

    train_df = read_csv(train)
    dev_df = read_csv(dev)
    test_df = read_csv(test)

    #docs = train_df.text.tolist() + dev_df.text.tolist() + test_df.text.tolist()
    docs = train_df.text.tolist()
    labels = train_df.label.tolist()
    docs = [preprocess(doc) for doc in docs]

    #docs = docs
    #labels = labels
    #docs = train_df.text.tolist() + dev_df.text.tolist() + test_df.text.tolist()
    #labels = train_df.label.tolist() + dev_df.label.tolist() + test_df.label.tolist()

    # Only use the vocabulary from the training data
    tfidf_mtx, vocab, tfidf_vectorizer = get_tfidf(docs)
    df = {'word_id':[], 'word':[]}
    raw_vocab = vocab
    vocab, metamap_map = metamap_normalize(metamap_dict, vocab)
    word_idx = {}
    for i, w in enumerate(vocab):
        word_id = len(docs) + i
        df['word_id'].append(word_id)
        df['word'].append(w)
        word_idx[w] = word_id
    pd.DataFrame(df).to_csv('{}/word.ids'.format(data_path), index=False)

    # Create word-word edges
    wc_map, _ = get_count(docs)
    total_count = sum(wc_map.values())
    wc_map = normalize_wc_map(wc_map, metamap_map)
    #vocab = set(vocab)
    with open('{}/train.csv.parsed.norm'.format(data_path), 'r') as fr:
        with open('{}/word_word.edges'.format(data_path), 'w') as fw:
            for line in fr:
                w1, w2, _, pair_count = line.strip().split('\t')
                pair_count = int(pair_count)
                if w1 not in wc_map or w1 not in vocab:
                    #print('not found: {}'.format(w1))
                    continue
                if w2 not in wc_map or w2 not in vocab:
                    #print('not found: {}'.format(w2))
                    continue
                pmi = get_pmi(pair_count, wc_map[w1], wc_map[w2], total_count)
                fw.write('{}\t{}\t{}\n'.format(word_idx[w1], word_idx[w2], pmi))

    # Create nodes and edges for all documents
    df = {'doc_id':[], 'text':[], 'label':[]}
    for i, (doc, label) in enumerate(zip(docs, labels)):
        df['doc_id'].append(i)
        df['text'].append(doc)
        df['label'].append(int(label))
    pd.DataFrame(df).to_csv('{}/doc.ids'.format(data_path), index=False)

    with open('{}/doc_word.edges'.format(data_path), 'w') as fw:
        all_tfidf_mtx = tfidf_vectorizer.transform(docs)
        nonzeros = find(all_tfidf_mtx)
        doc2norm = {}
        for i, j, val in zip(nonzeros[0], nonzeros[1], nonzeros[2]):
            word = raw_vocab[j]
            if word not in metamap_map:
                continue
            norm = metamap_map[word]
            word_id = word_idx[norm]
            if i not in doc2norm:
                doc2norm[i] = {}
            else:
                if word_id not in doc2norm[i]:
                    doc2norm[i][word_id] = []
                doc2norm[i][word_id].append(val)

        for doc_id, word2vals in doc2norm.items():
            for word_id, vals in word2vals.items():
                val = sum(vals)/len(vals)
                fw.write('{}\t{}\t{}\n'.format(doc_id, word_id, val))
                fw.write('{}\t{}\t{}\n'.format(word_id, doc_id, val))
            fw.write('{}\t{}\t{}\n'.format(doc_id, doc_id, 1))


    #with open('{}/data.split'.format(data_path), 'w') as fw:
    #    fw.write('{}\t{}\t{}\n'.format(len(train_df), len(dev_df), len(test_df)))

    #word2vec_initialization(data_path, metamap_map)
    # encode word nodes
    normed_word_embeds = {}
    for i, word in enumerate(raw_vocab):
        if word not in metamap_map:
            continue
        word_embedding = get_word_embedding(i, word)
        norm = metamap_map[word]
        word_id = word_idx[norm]
        if word_id not in normed_word_embeds:
            normed_word_embeds[word_id] = []
        normed_word_embeds[word_id].append(word_embedding)

    with open('{}/word.nodes'.format(data_path), 'w') as fw:
        for word_id, embeds in normed_word_embeds.items():
            word_embedding = np.asarray(embeds).mean(axis=0).tolist()
            fw.write('{}\t{}\n'.format(word_id, '\t'.join([str(x) for x in word_embedding])))

    with open('{}/doc.nodes'.format(data_path), 'w') as fw:
        for i, (doc_id, text, label) in enumerate(zip(df['doc_id'], df['text'], df['label'])):
            doc_embedding = get_doc_embedding(text, metamap_map)
            fw.write('{}\t{}\t{}\n'.format(doc_id, '\t'.join([str(x) for x in doc_embedding]), label))
        fw.close()



