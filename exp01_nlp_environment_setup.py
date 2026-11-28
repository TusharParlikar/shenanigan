"""
Experiment 1: Install and configure a Python NLP environment (NLTK, SpaCy,
Gensim, Hugging Face Transformers) and verify each library with a sample task.

Install:
    pip install nltk spacy gensim transformers torch pandas
    python -m spacy download en_core_web_sm

CONCEPTS
- An NLP environment combines libraries for text processing (NLTK), fast
  production pipelines (SpaCy), word representations (Gensim) and pre-trained
  deep learning models (Transformers).
- Most libraries need separately downloaded resources (corpora, language
  models, model weights), so an install is only "verified" once a sample task
  actually runs.
- Word embeddings: words used in similar contexts get similar dense vectors.
- Transformers use self-attention to build context-dependent representations;
  the pipeline() API hides tokenization + model + post-processing.
"""

import nltk
import spacy
import gensim
import transformers
import pandas as pd

# NLTK ships code only; corpora/models are downloaded on demand.
for resource in ['punkt', 'punkt_tab', 'stopwords',
                 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    nltk.download(resource, quiet=True)
print('Libraries installed and NLTK resources downloaded successfully.')

pd.set_option('display.max_colwidth', None)
versions = pd.DataFrame({
    'Library': ['NLTK', 'SpaCy', 'Gensim', 'Transformers'],
    'Version': [nltk.__version__, spacy.__version__, gensim.__version__, transformers.__version__],
})
print(versions)

# ---------------------------------------------------------------------------
# Step 3: NLTK - tokenization, stop-word removal, POS tagging
# Tokenization  = splitting text into sentences / words (tokens).
# Stop words    = very frequent function words (the, is, and) with little meaning.
# POS tagging   = labelling each token with its part of speech (Penn Treebank tags).
# ---------------------------------------------------------------------------
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

text = ("Natural Language Processing helps computers understand human language. "
        "Python provides powerful libraries for NLP tasks.")

sentences = sent_tokenize(text)
tokens = word_tokenize(text)

stop_words = set(stopwords.words('english'))
filtered = [w for w in tokens if w.isalpha() and w.lower() not in stop_words]

pos_tags = nltk.pos_tag(tokens)  # averaged perceptron tagger under the hood

print('Sentences:\n', sentences)
print('\nWord Tokens:\n', tokens)
print('\nAfter Stop-word Removal:\n', filtered)
print('\nPOS Tags:\n', pos_tags)

# ---------------------------------------------------------------------------
# Step 4: SpaCy - linguistic annotation in ONE pass
# nlp(text) runs the whole pipeline: tokenizer -> tagger -> parser -> lemmatizer -> NER.
# lemma_ = dictionary base form, pos_ = universal POS, dep_ = dependency relation.
# ---------------------------------------------------------------------------
nlp = spacy.load('en_core_web_sm')
doc = nlp("Sundar Pichai is the CEO of Google, which opened a new office in Pune in 2024.")
print('\nPipeline components:', nlp.pipe_names)

spacy_df = pd.DataFrame(
    [(t.text, t.lemma_, t.pos_, t.dep_, t.is_stop) for t in doc],
    columns=['Token', 'Lemma', 'POS', 'Dependency', 'Is_Stopword']
)
print(spacy_df)

# Step 5: Named Entity Recognition - spans labelled PERSON, ORG, GPE, DATE...
ner_df = pd.DataFrame(
    [(ent.text, ent.label_, spacy.explain(ent.label_)) for ent in doc.ents],
    columns=['Entity', 'Label', 'Description']
)
print(ner_df)

# ---------------------------------------------------------------------------
# Step 6: Gensim - training Word2Vec
# Word2Vec learns a dense vector per word by predicting neighbouring words
# within a `window`. vector_size = embedding dimension.
# NOTE: on a 6-sentence corpus the similarities only prove the API works,
# not real word meaning.
# ---------------------------------------------------------------------------
from gensim.models import Word2Vec

corpus = [
    "natural language processing is a field of artificial intelligence",
    "machine learning models learn patterns from data",
    "deep learning is a subset of machine learning",
    "word embeddings represent words as dense vectors",
    "language models predict the next word in a sentence",
    "artificial intelligence includes machine learning and deep learning",
]
tokenized_corpus = [sentence.split() for sentence in corpus]

w2v = Word2Vec(sentences=tokenized_corpus, vector_size=50, window=3,
               min_count=1, workers=1, epochs=100, seed=42)

print('\nVocabulary size:', len(w2v.wv))
print('Vector shape for "learning":', w2v.wv['learning'].shape)
print('First 10 values:', w2v.wv['learning'][:10])
print('\nWords most similar to "learning":')
for word, score in w2v.wv.most_similar('learning', topn=5):  # cosine similarity
    print(f'  {word:15s} {score:.4f}')

# ---------------------------------------------------------------------------
# Step 7: Transformers - pre-trained sentiment analysis
# DistilBERT fine-tuned on SST-2: no training needed, transfer learning.
# ---------------------------------------------------------------------------
from transformers import pipeline

classifier = pipeline('sentiment-analysis',
                      model='distilbert-base-uncased-finetuned-sst-2-english')

samples = [
    "I love working with natural language processing libraries!",
    "The installation process was slow and frustrating.",
    "Transformers make NLP tasks much easier to implement.",
]
preds = classifier(samples)

hf_df = pd.DataFrame({
    'Text': samples,
    'Label': [p['label'] for p in preds],
    'Score': [round(p['score'], 4) for p in preds],
})
print(hf_df)

# Step 8: Installation verification summary
summary = pd.DataFrame({
    'Library': ['NLTK', 'SpaCy', 'Gensim', 'Transformers'],
    'Version': [nltk.__version__, spacy.__version__, gensim.__version__, transformers.__version__],
    'Sample Task Performed': ['Tokenization, Stop-words, POS Tagging',
                              'Lemmatization, Dependency Parsing, NER',
                              'Word2Vec Training & Similarity',
                              'Sentiment Analysis (DistilBERT)'],
    'Status': ['Verified'] * 4,
})
print(summary)
