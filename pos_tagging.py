import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer

corpus = """
In the heart of the Pacific Ocean, there existed a mysterious island known as Wonders. This island was shrouded in mist and legend, drawing adventurers and scientists alike to its shores. The island was home to a diverse array of flora and fauna, each more fascinating than the last.
One sunny day, a group of explorers led by the intrepid Captain James set out to uncover the secrets of Wonders. Their ship, the "Sea Dragon," sailed through calm waters and turbulent storms, finally reaching the island's shores after weeks at sea. As they disembarked, they were greeted by the sweet songs of exotic birds and the gentle rustle of leaves in the breeze.
Captain James, a man of great courage and curiosity, led his team into the dense jungle that covered most of the island. They trekked through thick underbrush, crossing narrow streams and climbing steep hills. Along the way, they encountered creatures they had never seen before: a bird with iridescent feathers, a snake with scales that shimmered like diamonds, and even a small mammal that seemed to defy gravity.
As they ventured deeper into the jungle, they stumbled upon an ancient temple hidden behind a waterfall. The temple was adorned with intricate carvings and mysterious symbols that told stories of a long-lost civilization. Dr. Maria, a brilliant archaeologist, was thrilled to discover these artifacts and began to decipher their meanings.
Inside the temple, they found a series of cryptic messages etched into the walls. These messages spoke of a hidden treasure buried somewhere on the island. The team was determined to find this treasure, and their search led them to a hidden cave deep within the island's volcanic core.
As they navigated the cave, they encountered numerous obstacles: narrow tunnels, treacherous paths, and even a hidden lake. But their perseverance paid off when they finally reached a large chamber filled with glittering jewels and ancient artifacts.
Among the treasures, they found a journal belonging to a previous explorer who had visited the island centuries ago. The journal told the story of a magical spring that granted eternal youth to those who drank from it. The team was skeptical but decided to search for the spring nonetheless.
After days of searching, they finally found the spring in a secluded valley. The water was crystal clear, and as they drank from it, they felt a strange sensation wash over them. Whether it was the magic of the spring or the sheer wonder of their discovery, they knew that their lives would never be the same.
As they prepared to leave the island, Captain James reflected on their journey. "This island," he said, "is a reminder that there is still so much to discover in this world. And sometimes, the greatest wonders are those we least expect."
"""

sns = SnowballStemmer("english")
sentences = sent_tokenize(corpus)
stopwords = nltk.corpus.stopwords.words("english")  

# nltk.download("averaged_perceptron_tagger")

for i in range(len(sentences)):
    words = word_tokenize(sentences[i])
    words = [sns.stem(word) for word in words if word.isalpha() and word not in set(stopwords)] 
    # sentences[i] = " ".join(words)
    pos_tags = nltk.pos_tag(words)
    print(pos_tags)

# print(sentences)


print(nltk.pos_tag("I am Learning NLP".split()))