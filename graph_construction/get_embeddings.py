from gensim.models import Word2Vec
from gensim.utils import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

word2vec_model = Word2Vec.load("tw_word2vec.model")

random_word_embeddings = {}

def get_word_embedding(word_id, word):
    try:
        vector = word2vec_model.wv[word]
    except:
        emb_size = 768
        np.random.seed(int(word_id))
        vector = np.random.rand(1, emb_size)[0].tolist()
        random_word_embeddings[word] = vector
    return vector

def get_doc_embedding(doc):
    words = tokenize(doc)
    vectors = []
    for word in words:
        try:
            vectors.append(get_word_embedding(-1, word))
        except:
            if word in random_word_embeddings:
                vectors.append(random_word_embeddings[word])
    return np.asarray(vectors).mean(axis=0)/len(vectors)

if __name__ == '__main__':
    #print(get_word_embedding('i'))
    print(get_doc_embedding('i am so happy today'))

