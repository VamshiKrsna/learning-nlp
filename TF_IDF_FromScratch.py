import numpy as np
from collections import Counter
import math

def compute_tf(sentence):
    words = sentence.split()
    total_words = len(words)
    word_count = Counter(words)
    tf = {word: count / total_words for word, count in word_count.items()}
    return tf

def compute_idf(corpus):
    total_docs = len(corpus)
    vocab = set(word for sentence in corpus for word in sentence.split())
    idf = {}
    for word in vocab:
        doc_count = sum(1 for sentence in corpus if word in sentence.split())
        idf[word] = math.log(total_docs / (1 + doc_count))  # Adding 1 to avoid division by zero
    return idf

def compute_tfidf(corpus):
    idf = compute_idf(corpus)
    tfidf_vectors = []
    
    for sentence in corpus:
        tf = compute_tf(sentence)
        tfidf = {word: tf[word] * idf[word] for word in tf}
        tfidf_vectors.append(tfidf)
    
    return tfidf_vectors

# Example
corpus = ["I love NLP", "NLP is fun", "I love learning"]
tfidf_vectors = compute_tfidf(corpus)

print("TF-IDF Vectors:")
for vec in tfidf_vectors:
    print(vec)

"""
TF-IDF Vectors:
{'I': 0.0, 'love': 0.0, 'NLP': 0.0}
{'NLP': 0.0, 'is': 0.13515503603605478, 'fun': 0.13515503603605478}
{'I': 0.0, 'love': 0.0, 'learning': 0.13515503603605478}
"""