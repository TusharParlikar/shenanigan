"""
Experiment 2: Text preprocessing pipeline - normalization, tokenization,
stop-word removal, stemming and lemmatization.

Install:
    pip install nltk pandas numpy

CONCEPTS
- Raw text has noise and inconsistent formatting; many surface forms of the
  same word ("NLP", "Nlp.", "nlp"). Preprocessing reduces that variation.
- Normalization : lowercasing, removing punctuation/digits/extra spaces.
- Tokenization  : splitting text into words (tokens).
- Stop words    : dropping frequent function words (the, is, and). Careful:
  NLTK's list contains "not", which matters for sentiment analysis.
- Stemming      : heuristic suffix stripping (Porter). Fast, but can output
  non-words (studies -> studi).
- Lemmatization : dictionary-based base form (WordNet) (studies -> study,
  was -> be). Needs the word's POS; by default every word is a noun.

    Stemming                     | Lemmatization
    Rule-based suffix stripping  | Dictionary + morphology based
    May output non-words (studi) | Always real words (study)
    Fast, ignores POS            | Slower, more accurate with POS
"""

import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4',
                 'averaged_perceptron_tagger_eng']:
    nltk.download(resource, quiet=True)
print('NLTK resources downloaded successfully.')

pd.set_option('display.max_colwidth', None)

# Step 2: Sample raw documents (note the digits, '&', '!!', mixed case)
documents = [
    'Natural Language Processing is transforming Artificial Intelligence.',
    'Machine learning models require clean and normalized text data.',
    'Tokenization, stemming, and lemmatization are important NLP preprocessing steps.',
    'The researchers were studying 3 different datasets & running better experiments!!',
    'A language model learns patterns, and models trained on studies of languages learned faster.',
]
for i, doc in enumerate(documents, 1):
    print(f'Document {i}: {doc}')

# Step 3: Initialize NLP tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
print('Number of English stop words:', len(stop_words))
print('Sample stop words:', sorted(stop_words)[:15])


# Step 4: The preprocessing pipeline
def get_wordnet_pos(tag):
    # Map Penn Treebank POS tags (JJ, VB, RB, NN...) to WordNet POS constants.
    # The lemmatizer needs this, otherwise "studying" stays "studying".
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN


def preprocess_text(text):
    # Normalization: lowercase, keep only letters, collapse whitespace
    normalized = text.lower()
    normalized = re.sub(r'[^a-z\s]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    # Tokenization
    tokens = word_tokenize(normalized)

    # Stop-word removal
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Stemming (Porter rules)
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

    # Lemmatization (default: every word treated as a noun)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # Lemmatization with POS tags -> correct base forms for verbs/adjectives
    pos_tags = nltk.pos_tag(filtered_tokens)
    pos_lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]

    return {
        'Original_Text': text,
        'Normalized_Text': normalized,
        'Tokens': tokens,
        'After_Stopword_Removal': filtered_tokens,
        'Stemmed': stemmed_tokens,
        'Lemmatized': lemmatized_tokens,
        'POS_Lemmatized': pos_lemmatized,
        'Clean_Text': ' '.join(pos_lemmatized),
    }


# Step 5: Apply the pipeline to all documents
results = [preprocess_text(doc) for doc in documents]
df = pd.DataFrame(results)
print(df[['Original_Text', 'Normalized_Text', 'Tokens', 'After_Stopword_Removal']])
print(df[['Stemmed', 'Lemmatized', 'POS_Lemmatized', 'Clean_Text']])

# Step 6: Stemming vs Lemmatization - irregular forms (mice, geese, was, better)
# expose the difference: only the POS-aware lemmatizer gets them right.
word_pos = [('studies', 'n'), ('studying', 'v'), ('running', 'v'), ('better', 'a'),
            ('was', 'v'), ('normalized', 'v'), ('transforming', 'v'),
            ('experiments', 'n'), ('mice', 'n'), ('geese', 'n')]

comparison = pd.DataFrame({
    'Word': [w for w, p in word_pos],
    'POS': [p for w, p in word_pos],
    'Porter_Stem': [stemmer.stem(w) for w, p in word_pos],
    'Lemma_(noun_default)': [lemmatizer.lemmatize(w) for w, p in word_pos],
    'Lemma_(with_POS)': [lemmatizer.lemmatize(w, p) for w, p in word_pos],
})
print(comparison)


# Step 7: Effect of preprocessing on vocabulary size.
# Each stage merges surface forms, so the number of unique tokens shrinks.
def vocab(column):
    return len(set(tok for row in df[column] for tok in row))


raw_vocab = len(set(tok for doc in documents for tok in word_tokenize(doc)))

stats = pd.DataFrame({
    'Stage': ['Raw tokens', 'After normalization', 'After stop-word removal',
              'After stemming', 'After lemmatization (POS-aware)'],
    'Vocabulary_Size': [raw_vocab, vocab('Tokens'), vocab('After_Stopword_Removal'),
                        vocab('Stemmed'), vocab('POS_Lemmatized')],
})
print(stats)  # expected: 50 -> 44 -> 36 -> 31 -> 32
