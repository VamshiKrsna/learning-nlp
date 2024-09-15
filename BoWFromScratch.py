import numpy as np
import re 

class BagOfWords:
    def __init__(self):
        self.vocab = {}

    def tokenize(self,text):
        text = re.sub(r"[^\w\s]", "", text.lower())
        return text.split()

    def fit(self,corpus):
        idx = 0
        for sentence in corpus:
            for word in self.tokenize(sentence):
                if word not in self.vocab:
                    self.vocab[word] = idx
                    idx += 1
    
    def transform(self,corpus):
        bow_matrix = np.zeros((len(corpus),len(self.vocab)),dtype = int)
        for i, sentence in enumerate(corpus):
            for word in self.tokenize(sentence):
                if word in self.vocab:
                    bow_matrix[i,self.vocab[word]] += 1
        return bow_matrix
    
    def fit_transform(self,corpus):
        self.fit(corpus)
        return self.transform(corpus)
    

# Example : 
corpus = [
    "He is a good boy",
    "She is a good girl",
    "Boy and girl are good"
]

bow = BagOfWords()
bow_matrix = bow.fit_transform(corpus)

print("Vocabulary : ", bow.vocab)
print("Bag Of Words Matrix : \n", bow_matrix)