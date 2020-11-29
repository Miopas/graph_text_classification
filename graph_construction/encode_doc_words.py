import sys
import pandas as pd
from str_utils import read_csv, preprocess
import math

#from spacy.tokenizer import Tokenizer
#from spacy.lang.en import English
#nlp = English()
## Create a blank Tokenizer with just the English vocab
#tokenizer = Tokenizer(nlp.vocab)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import find
from sklearn.feature_extraction import text

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
    vocab = list(vocab)
    word_idx = {}
    for i, w in enumerate(vocab):
        word_id = len(docs) + i
        df['word_id'].append(word_id)
        df['word'].append(w)
        word_idx[w] = word_id
    pd.DataFrame(df).to_csv('{}/word.ids'.format(data_path), index=False)

    # Create word-word edges
    wc_map, vocab2 = get_count(docs)
    total_count = sum(wc_map.values())
    #vocab = set(vocab)
    with open('{}/train.csv.parsed'.format(data_path), 'r') as fr:
        with open('{}/word_word.edges'.format(data_path), 'w') as fw:
            for line in fr:
                w1, w2, _, pair_count = line.strip().split('\t')
                pair_count = int(pair_count)
                if w1 not in wc_map or w1 not in vocab:
                    print('not found: {}'.format(w1))
                    continue
                if w2 not in wc_map or w2 not in vocab:
                    print('not found: {}'.format(w2))
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
        #nonzeros = find(tfidf_mtx)
        #for i, j, val in zip(nonzeros[0], len(docs)+nonzeros[1], nonzeros[2]):
        #    fw.write('{}\t{}\t{}\n'.format(i, j, val))
        all_tfidf_mtx = tfidf_vectorizer.transform(docs)
        nonzeros = find(all_tfidf_mtx)
        for i, j, val in zip(nonzeros[0], len(docs)+nonzeros[1], nonzeros[2]):
            fw.write('{}\t{}\t{}\n'.format(i, j, val))

    with open('{}/data.split'.format(data_path), 'w') as fw:
        fw.write('{}\t{}\t{}\n'.format(len(train_df), len(dev_df), len(test_df)))


