# Lemmatization is similar to stemming, 
# But, Lemmatization derives lemma (root) of a word, not just a stem.
import nltk
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
print(wnl.lemmatize("Dogs",pos = "n"))
print(wnl.lemmatize("Eating",pos = "v"))