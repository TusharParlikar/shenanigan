"""
Experiment 4: Construct Bag-of-Words (BoW) and N-Gram models from a corpus and
analyze the frequency distribution of words and phrases.

Install:
    pip install nltk scikit-learn pandas numpy matplotlib

CONCEPTS
- Bag-of-Words: a document = counts of the words it contains, ORDER IGNORED.
  Build a vocabulary V of unique words; each document becomes a vector of
  length |V|; stacking them gives the Document-Term Matrix (DTM).
  Limits: loses order ("dog bites man" == "man bites dog"), sparse, no meaning.
- N-grams: contiguous sequences of N tokens - unigram (natural), bigram
  (natural language), trigram (natural language processing). A sentence of
  T tokens has T - N + 1 N-grams. They keep local order / phrases, but the
  vocabulary and sparsity grow fast with N.
- Zipf's law: frequency of a word is inversely proportional to its rank,
  f(r) ~ 1/r, so log(freq) vs log(rank) is a line with slope about -1.
  A few words cover a large share of all text; most words are rare.
- Hapax legomena: words that occur exactly once.
"""

import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, gutenberg
from nltk.util import ngrams
from nltk import FreqDist
from sklearn.feature_extraction.text import CountVectorizer

for resource in ['punkt', 'punkt_tab', 'stopwords', 'gutenberg']:
    nltk.download(resource, quiet=True)
print('NLTK resources downloaded successfully.')

pd.set_option('display.max_colwidth', None)

# Step 2: Corpus
corpus = [
    'Natural language processing enables computers to understand human language.',
    'Machine learning models learn patterns from large text data.',
    'Deep learning has improved natural language processing tasks such as translation.',
    'Text classification and sentiment analysis are common natural language processing applications.',
    'Machine learning algorithms require numerical features extracted from text data.',
    'The bag of words model represents text data as word frequency vectors.',
    'N-gram models capture the order of words in natural language text.',
    'Deep learning and machine learning are subfields of artificial intelligence.',
]
for i, doc in enumerate(corpus, 1):
    print(f'D{i}: {doc}')

# Step 3: Preprocess (lowercase, letters only, tokenize, drop stop words)
stop_words = set(stopwords.words('english'))


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)  # remove punctuation and digits
    tokens = word_tokenize(text)
    return [t for t in tokens if t not in stop_words and len(t) > 1]


tokenized = [preprocess(doc) for doc in corpus]
prep_df = pd.DataFrame({'Document': [f'D{i}' for i in range(1, len(corpus) + 1)],
                        'Tokens': tokenized,
                        'Token_Count': [len(t) for t in tokenized]})
print(prep_df)

# Step 4: Bag-of-Words FROM SCRATCH
vocabulary = sorted(set(word for doc in tokenized for word in doc))  # vocabulary V
word_index = {word: i for i, word in enumerate(vocabulary)}          # word -> column
print('Vocabulary size:', len(vocabulary))
print('Vocabulary:', vocabulary)

# DTM[d, j] = how many times vocabulary word j occurs in document d
bow_manual = np.zeros((len(tokenized), len(vocabulary)), dtype=int)
for d, doc in enumerate(tokenized):
    for word in doc:
        bow_manual[d, word_index[word]] += 1
print('\nDocument-term matrix shape:', bow_manual.shape)

dtm = pd.DataFrame(bow_manual, columns=vocabulary,
                   index=[f'D{i}' for i in range(1, len(corpus) + 1)])
top_cols = dtm.sum().sort_values(ascending=False).index[:12]
print(dtm[top_cols])

# Step 5: Same BoW with scikit-learn, verify, and measure sparsity
clean_docs = [' '.join(doc) for doc in tokenized]
vectorizer = CountVectorizer(analyzer=str.split)
bow_sklearn = vectorizer.fit_transform(clean_docs)  # returns a SPARSE matrix

print('Scikit-learn vocabulary size:', len(vectorizer.get_feature_names_out()))
print('Same vocabulary as manual model:', list(vectorizer.get_feature_names_out()) == vocabulary)
print('Same document-term matrix as manual model:', np.array_equal(bow_sklearn.toarray(), bow_manual))

# Sparsity = share of zero cells; text matrices are mostly zeros
total_cells = bow_manual.size
zero_cells = (bow_manual == 0).sum()
print(f'\nSparsity: {zero_cells} of {total_cells} entries are zero ({zero_cells / total_cells:.1%})')

# Step 6: Unigram frequency distribution
BAR_COLOR = '#2a78d6'
INK = '#333333'


def plot_top(items, title, xlabel='Frequency'):
    labels = [' '.join(k) if isinstance(k, tuple) else k for k, _ in items][::-1]
    values = [v for _, v in items][::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(items) + 1))
    bars = ax.barh(labels, values, color=BAR_COLOR, height=0.6)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2, str(v),
                va='center', fontsize=9, color=INK)
    ax.set_title(title, loc='left', fontsize=12, color=INK)
    ax.set_xlabel(xlabel, color=INK)
    ax.tick_params(colors=INK)
    ax.xaxis.grid(True, color='#e5e5e5')
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.set_xlim(0, max(values) * 1.15)
    plt.tight_layout()
    plt.show()


all_tokens = [word for doc in tokenized for word in doc]
unigram_fd = FreqDist(all_tokens)
print('Total tokens:', unigram_fd.N())     # N = total count
print('Unique unigrams:', unigram_fd.B())  # B = number of bins (types)
print('Words occurring only once (hapax legomena):', len(unigram_fd.hapaxes()))
plot_top(unigram_fd.most_common(12), 'Top 12 unigrams in the corpus')

# Step 7: Bigrams and trigrams.
# N-grams are built INSIDE each document so they never cross document boundaries.
bigrams = [bg for doc in tokenized for bg in ngrams(doc, 2)]
trigrams = [tg for doc in tokenized for tg in ngrams(doc, 3)]
bigram_fd = FreqDist(bigrams)
trigram_fd = FreqDist(trigrams)
print('Example bigrams from D1:', list(ngrams(tokenized[0], 2)))
print('Example trigrams from D1:', list(ngrams(tokenized[0], 3)))
plot_top(bigram_fd.most_common(10), 'Top 10 bigrams in the corpus')

print(pd.DataFrame([(' '.join(t), c) for t, c in trigram_fd.most_common(8)],
                   columns=['Trigram', 'Frequency']))

# Step 8: Vocabulary size and sparsity per N-gram order.
# Higher N -> more features, each seen fewer times -> sparser matrix.
rows = []
for n_range, name in [((1, 1), 'Unigram'), ((2, 2), 'Bigram'), ((3, 3), 'Trigram'),
                      ((1, 2), 'Unigram + Bigram')]:
    cv = CountVectorizer(ngram_range=n_range, token_pattern=r'\S+')
    X = cv.fit_transform(clean_docs).toarray()
    counts = X.sum(axis=0)
    rows.append({'Model': name,
                 'Vocabulary_Size': X.shape[1],
                 'Features_Seen_Once': int((counts == 1).sum()),
                 'Sparsity_%': round(100 * (X == 0).mean(), 1)})
print(pd.DataFrame(rows))

# Step 9: Large corpus (Jane Austen's "Emma") and Zipf's law
raw_words = gutenberg.words('austen-emma.txt')
words = [w.lower() for w in raw_words if w.isalpha()]
emma_fd = FreqDist(words)
emma_content_fd = FreqDist(w for w in words if w not in stop_words)

print('Total words:', emma_fd.N())
print('Vocabulary size:', emma_fd.B())
print(f'Words occurring only once: {len(emma_fd.hapaxes())} '
      f'({len(emma_fd.hapaxes()) / emma_fd.B():.1%} of the vocabulary)')
top10_share = sum(c for _, c in emma_fd.most_common(10)) / emma_fd.N()
print(f'Share of all words covered by the 10 most frequent words: {top10_share:.1%}')

print(pd.DataFrame({'Top words (all)': [f'{w} ({c})' for w, c in emma_fd.most_common(10)],
                    'Top words (stop words removed)': [f'{w} ({c})' for w, c in
                                                       emma_content_fd.most_common(10)]}))

# Fit a line to log10(frequency) vs log10(rank): slope near -1 confirms Zipf
freqs = np.array(sorted(emma_fd.values(), reverse=True))
ranks = np.arange(1, len(freqs) + 1)
slope_all = np.polyfit(np.log10(ranks), np.log10(freqs), 1)[0]
slope_top, intercept_top = np.polyfit(np.log10(ranks[:1000]), np.log10(freqs[:1000]), 1)
print("Zipf's law predicts a log-log slope of about -1")
print(f'Fitted slope, top 1000 ranks: {slope_top:.2f}')
print(f'Fitted slope, all ranks:      {slope_all:.2f}')  # steeper due to long tail of rare words

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.loglog(ranks, freqs, color=BAR_COLOR, linewidth=2, label='Observed frequency')
ax.loglog(ranks, 10 ** intercept_top * ranks ** slope_top, color='#888888', linewidth=1.5,
          linestyle='--', label=f'Fitted line, top 1000 ranks (slope {slope_top:.2f})')
ax.set_title("Zipf's law: rank vs frequency in 'Emma'", loc='left', fontsize=12, color=INK)
ax.set_xlabel('Rank (log scale)', color=INK)
ax.set_ylabel('Frequency (log scale)', color=INK)
ax.tick_params(colors=INK)
ax.grid(True, which='major', color='#e5e5e5')
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False)
plt.tight_layout()
plt.show()

# N-gram vocabulary growth: unique N-grams explode and most are seen only once
growth = []
for n in (1, 2, 3):
    fd = FreqDist(ngrams(words, n))
    growth.append({'N': n, 'Unique_N-grams': fd.B(),
                   'Seen_Only_Once_%': round(100 * len(fd.hapaxes()) / fd.B(), 1)})
print(pd.DataFrame(growth))
