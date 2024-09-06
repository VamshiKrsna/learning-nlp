from nltk.stem import PorterStemmer,RegexpStemmer,SnowballStemmer
ps = PorterStemmer()

example_words = ["python", "pythoner", "pythoning", "eating","eatery","drinkery","drinker","drinking" ,"pythoned", "pythonly", "pythonista", "pythoneer"]

for w in example_words:
    print(f"{w} ---> {ps.stem(w)}")

# Stemming is not the best practice as it changes meaning of some words.

print(ps.stem("Setter"))

rs = RegexpStemmer('ing$|s$|e$|able$|er$', min=4)
# We can define a regex to stem tokens
print(rs.stem("Setter"))
print(rs.stem("Desireable"))

ss = SnowballStemmer("english")
print(ss.stem("Desireable"))
print(ss.stem("lively"))
