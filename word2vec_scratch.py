import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import numpy as np
from collections import Counter
import torch 
import torch.nn as nn
import torch.optim as optim 

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

corpus = """
In the kingdom, the king and queen ruled with kindness and wisdom. One day, a brave man named Jack came to the palace to request the queen's help. His wife, a woman named Sarah, was ill and needed the queen's special healing powers. The queen, being a just and compassionate ruler, agreed to help. She used her magic to cure Sarah's illness, and soon the woman was back on her feet. The king, pleased with the queen's good deed, rewarded Jack with a noble title and a grand feast was held in their honor. As the man and woman danced together, the king and queen smiled, happy to have brought joy to their loyal subjects.
"""

tokens = word_tokenize(corpus)

# Eliminate punctuations, stopwords, and lemmatize
table = str.maketrans('', '', string.punctuation)
tokens = [w.translate(table) for w in tokens]

stop_words = set(stopwords.words('english'))
tokens = [w for w in tokens if w.lower() not in stop_words]

lemmatizer = WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(w) for w in tokens]

vocab = set(tokens)
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}
vocab_size = len(vocab)

class Word2Vec(nn.Module):  
    def __init__(self, vocab_size, embedding_dim):
        """
        Initialize the word2vec model.
        """
        super(Word2Vec, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        """
        Forward Pass for the model.
        """
        embeds = self.embedding(inputs).sum(dim=0)
        out = self.output_layer(embeds)
        return out
    
    def get_embedding(self,word_idx):
        """
        Get the embedding for a specific word.
        """
        return self.embedding(word_idx)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
def train(model,pairs,epochs,lr):
    """
    Training the model using Stochastic Gradient Descent (SGD).
    """
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        for target_word, context_word in pairs:
            target = torch.tensor([target_word], dtype=torch.long)
            context = torch.tensor([context_word], dtype=torch.long)
            optimizer.zero_grad()
            output = model(target)
            output = output.unsqueeze(0)
            loss = criterion(output,context)
            loss.backward() # Back propagation
            optimizer.step() # Gradient Descent
            total_loss += loss.item()

        if epoch % 100 == 0:
            print("Epoch: {} Loss: {:.4f}".format(epoch, total_loss))

# Generate pairs
def generate_pairs(corpus,window_size):
    """
    Generates (target, context) word pairs of window_size from the corpus.
    """
    pairs = []
    for i, target_word in enumerate(corpus):    
        target_idx = word_to_idx[target_word]
        start = max(0, i - window_size)
        end = min(len(corpus), i + window_size + 1)
        context_words = [word_to_idx[corpus[j]] for j in range(start, end) if j != i]
        for context in context_words:
            pairs.append((target_idx, context))
    return pairs

window_size = 2 
pairs = generate_pairs(tokens,window_size)

# Training
embedding_dim = 3 # 3d Embeddings
epochs = 2000 # After 2000 epochs, the loss diminishes very slowly.
lr = 0.01
model = Word2Vec(vocab_size, embedding_dim)
train(model,pairs,epochs,lr)

def closest_word(embedding, word_embeddings):
    """
    Finding the closest words using Cosine Similarity.
    """
    similarities = []
    for i in range(len(word_embeddings)):  
        word_vec = word_embeddings[i].unsqueeze(0) 
        similarity = torch.cosine_similarity(embedding, word_vec)
        similarities.append(similarity)
    most_similar_idx = torch.argmax(torch.tensor(similarities)).item()
    return idx_to_word[most_similar_idx]

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_embeddings(model, vocab):
    embeddings = model.embedding.weight.detach().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, word in enumerate(vocab):
        ax.scatter(embeddings[i, 0], embeddings[i, 1], embeddings[i, 2], label=word)
    ax.legend()
    plt.show()

# Testing
king_embed = model.get_embedding(torch.tensor([word_to_idx['king']]))
man_embed = model.get_embedding(torch.tensor([word_to_idx['man']]))
queen_embed = model.get_embedding(torch.tensor([word_to_idx['queen']]))
king_man_queen_analogy = king_embed - man_embed + queen_embed

closest_word_to_analogy = closest_word(king_man_queen_analogy, model.embedding.weight)
print(f"King - Man + Queen = {closest_word_to_analogy}")

plot_embeddings(model, vocab)

"""
Output : 
Epoch: 0 Loss: 1319.4719
Epoch: 100 Loss: 992.7136
Epoch: 200 Loss: 918.5685
Epoch: 300 Loss: 899.7912
Epoch: 400 Loss: 889.5747
Epoch: 500 Loss: 882.3080
Epoch: 600 Loss: 877.4613
Epoch: 700 Loss: 874.0787
Epoch: 800 Loss: 871.4103
Epoch: 900 Loss: 869.0592
Epoch: 1000 Loss: 866.8461
Epoch: 1100 Loss: 864.7321
Epoch: 1200 Loss: 862.7804
Epoch: 1300 Loss: 861.0819
Epoch: 1400 Loss: 859.7139
Epoch: 1500 Loss: 858.6775
Epoch: 1600 Loss: 857.8783
Epoch: 1700 Loss: 857.2309
Epoch: 1800 Loss: 856.6844
Epoch: 1900 Loss: 856.2074
King - Man + Queen = queen # Should be woman, due to little training and corpus, it is a common inaccuracy.
"""