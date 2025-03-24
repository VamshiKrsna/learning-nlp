import nltk 
from nltk.corpus import reuters
from nltk import bigrams, trigrams 
from collections import Counter, defaultdict

nltk.download('reuters')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')


model = defaultdict(lambda: defaultdict(lambda: 0))

for sentence in reuters.sents():
    for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
        model[(w1, w2)][w3] += 1

for w1_w2 in model:
    total_count = float(sum(model[w1_w2].values()))
    
    for w3 in model[w1_w2]:
        model[w1_w2][w3] /= total_count
        # print(total_count)
        # print(model[w1_w2][w3])

# print(model)

sorted_probabilities = sorted(dict(model["the","price"]).items(), key=lambda x: x[1], reverse=True)

print("Most Probable Words following 'the price', in order:")
for word, prob in sorted_probabilities:
    print(f"{word}: {prob:.4f}")