import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

sentence = """
Apple is looking at buying U.K. startup for $1 billion. The News is very exciting.
"""

nltk.download("maxent_ne_chunker")

tokens = word_tokenize(sentence)
print("Tokens : ",tokens)
pos_tags = pos_tag(tokens)
print("Parts of speech : ",pos_tags)
named_entities = ne_chunk(pos_tags)
print("Named Entities : ",named_entities)

nltk.ne_chunk(pos_tags).draw()