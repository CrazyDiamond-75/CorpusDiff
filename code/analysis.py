import re
import pandas as pd
import spacy

# For spacy
nlp: spacy.Language

def do(funcs, x):
    for f in funcs:
        f(x)

def preprocess(text):
    text = text.lower()
    #text = re.sub(r"http\S+", " ", text)
    #text = re.sub(r"[^\w\s]", " ", text)
    #text = re.sub(r"\s+", " ", text).strip()
    
    #return [w.lemma_ for w in nlp(text)]
    #return [t.lemma_ if t.lemma_ != '--' else t.text for t in nlp(text)]
    return [t.lemma_ for t in nlp(text) if t.lemma_ != '--']
    #return nlp(text)

word_counts = {}
def count_words(line):
    global word_counts
    # Use a tokenizer to tokenize text.
    for word in preprocess(line):
        #print(word)
        if word not in word_counts:
            word_counts[word] = 1
        else:
            word_counts[word] += 1

def main():
    global nlp

    # German, core model, trained on news, small model (fastest).
    # Do `python -m spacy download de_core_web_sm` before running
    nlp = spacy.load("de_core_news_sm")

    print("LOADING DATASET", end="")
    df = pd.read_pickle('ndy_utf8_small.pkl')
    print("\rLOADING COMPLETE")

    #for i, row in df.iterrows():
    
    df['Text'].map(lambda x: do([count_words], x))
    print(word_counts.keys().sorted(key = lambda x : -word_counts[x]))

    print(df)


if __name__ == "__main__":
    main()

