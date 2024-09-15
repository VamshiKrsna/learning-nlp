from collections import defaultdict

class NGrams:
    def __init__(self, n = 2):
        self.n = n 
        self.ngram_dict = defaultdict(int)

    def generate_ngrams(self, text):
        words = text.split()
        ngrams = [tuple(words[i:i+self.n]) for i in range(len(words)-self.n+1)]
        return ngrams
    
    def fit(self, corpus):
        for sentence in corpus:
            for ngram in self.generate_ngrams(sentence):
                self.ngram_dict[ngram] += 1

    def get_ngrams(self):
        return dict(self.ngram_dict)
    

# Example :
corpus = [
    "He is a good boy",
    "She is a good girl",
    "Boy and girl are good"
]

ng = NGrams(n=2) # Bigrams
ng.fit(corpus)
print("Bi-Grams for given corpus : \n", ng.get_ngrams())

"""
Output :
Bi-Grams for given corpus : 
 {('He', 'is'): 1, ('is', 'a'): 2, ('a', 'good'): 2, ('good', 'boy'): 1, ('She', 'is'): 1, ('good', 'girl'): 1, ('Boy', 'and'): 1, ('and', 'girl'): 1, ('girl', 'are'): 1, ('are', 'good'): 1}
"""