import numpy as np

def one_hot_encoding(corpus):
    # Vocabulary
    vocab = sorted(set(word for sentence in corpus for word in sentence.split()))
    vocab_size = len(vocab)
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    
    # One hot encodings for each word in corpus
    one_hot_vectors = []
    for sentence in corpus:
        sentence_vector = []
        for word in sentence.split():
            vector = np.zeros(vocab_size)
            vector[word_to_index[word]] = 1
            sentence_vector.append(vector)
        one_hot_vectors.append(np.array(sentence_vector))
    
    return vocab, one_hot_vectors

# Example 
corpus = ["I love NLP", "NLP is fun", "I love learning"]
vocab, one_hot_vectors = one_hot_encoding(corpus)

print("Vocabulary:", vocab)
print("One-Hot Encoding:")
for vec in one_hot_vectors:
    print(vec)

"""
Output: 
Vocabulary: ['I', 'NLP', 'fun', 'is', 'learning', 'love']
One-Hot Encoding:
[[1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1.]
 [0. 1. 0. 0. 0. 0.]]
[[0. 1. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0.]
 [0. 0. 1. 0. 0. 0.]]
[[1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1.]
 [0. 0. 0. 0. 1. 0.]]
"""