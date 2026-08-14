import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from stopwordsiso import stopwords


# Set of German stop words for filtering
german_stopwords = stopwords("de")

df = pd.read_pickle("ndy_utf8_word_counts.pkl")
frequencies = dict(zip(df["Name"], df["Count"]))

for word in df["Name"]:
    if len(word) < 3 or word in german_stopwords:
        # Set frequency to 0 for words with length less than 3, which filters out most emojis.
        # Set frequency to 0 for German stop words.
        del frequencies[word]

wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(frequencies)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Word Counts")
plt.show()