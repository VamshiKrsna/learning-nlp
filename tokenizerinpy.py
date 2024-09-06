from nltk.tokenize import sent_tokenize, word_tokenize, line_tokenize, TreebankWordTokenizer
import nltk
# print(help(nltk))
corpus = """
Hello, I am Dobby. Dobby is a free elf.
Dobby was freed from the malfoys through harry's help.
Harry hid his sock in malfoy's book.
The book was handed to Dobby.
Implying that his master presented Dobby with clothes.
Hence Dobby was freed.
"""
sentences = sent_tokenize(corpus)
words = word_tokenize(corpus)
lines = line_tokenize(corpus)
print(f"Sentences : {sentences} \n No. : {len(sentences)}")
print(f"Words : {words} \n No. : {len(words)}")
print(f"Lines : {lines} \n No. : {len(lines)}")
print(f"Treebank Word Tokenizer : {TreebankWordTokenizer().tokenize(corpus)}")