"""
Experiment 5: Morphological Analysis and Part-of-Speech (POS) Tagging to
identify grammatical structures and linguistic properties of words.

Install:
    pip install nltk spacy pandas matplotlib
    python -m spacy download en_core_web_sm

CONCEPTS
- Morphology = internal structure of words. Morpheme = smallest meaningful
  unit: unhappiness = un- + happy + -ness.
  * Inflectional morphology adds grammatical features, same word
    (walk -> walked, cat -> cats).
  * Derivational morphology creates a NEW word, often a new class
    (happy -> happiness).
- POS tagging assigns a grammatical category to each word.
  * Penn Treebank tagset: 36 fine tags (NN, NNS, VBD, VBG, JJ ...).
  * Universal tagset: 12 coarse tags (NOUN, VERB, ADJ ...).
  * Main difficulty is AMBIGUITY: "I can can a can" -> modal, verb, noun.
- Tagging approaches (increasing context / features):
  * Default tagger  : one tag (NN) for every word.
  * Unigram tagger  : each word's most frequent tag in training data.
  * Bigram + backoff: also uses previous tag, falls back to unigram if unseen.
  * Averaged Perceptron: rich features of surrounding words (~97% on news).
- Taggers are evaluated by accuracy on a hand-tagged corpus (Penn Treebank).
"""

import random
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import map_tag, DefaultTagger, UnigramTagger, BigramTagger, PerceptronTagger
from nltk.corpus import treebank, wordnet

for resource in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger_eng', 'universal_tagset',
                 'wordnet', 'omw-1.4', 'treebank']:
    nltk.download(resource, quiet=True)
print('NLTK resources downloaded successfully.')

pd.set_option('display.max_colwidth', None)
nlp = spacy.load('en_core_web_sm')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# ---------------------------------------------------------------------------
# Step 3: Rule-based morphological analysis (affix detection)
# ---------------------------------------------------------------------------
PREFIXES = {'un': 'not', 're': 'again', 'mis': 'wrongly', 'dis': 'opposite',
            'pre': 'before', 'inter': 'between'}
SUFFIXES = {
    'ization': ('derivational', 'verb -> noun'),
    'ness': ('derivational', 'adjective -> noun'),
    'ment': ('derivational', 'verb -> noun'),
    'able': ('derivational', 'verb -> adjective'),
    'ly': ('derivational', 'adjective -> adverb'),
    'er': ('derivational/inflectional', 'agent or comparative'),
    'ing': ('inflectional', 'progressive / gerund'),
    'ed': ('inflectional', 'past tense'),
    'est': ('inflectional', 'superlative'),
    's': ('inflectional', 'plural / 3rd person'),
}


def analyze_affixes(word):
    w = word.lower()
    # Longest-match first. A prefix is accepted only if the remainder is a real
    # WordNet word (so "rest" is not re- + st). Still fooled by "reports" (re + ports).
    prefix = next((p for p in sorted(PREFIXES, key=len, reverse=True)
                   if w.startswith(p) and len(w) - len(p) >= 4
                   and wordnet.morphy(w[len(p):]) is not None), '')
    suffix = next((s for s in sorted(SUFFIXES, key=len, reverse=True)
                   if w.endswith(s) and len(w) - len(s) >= 3), '')
    return {
        'Word': word,
        'Prefix': f'{prefix}- ({PREFIXES[prefix]})' if prefix else '-',
        'Suffix': f'-{suffix}' if suffix else '-',
        'Suffix_Type': SUFFIXES[suffix][0] if suffix else '-',
        'Function': SUFFIXES[suffix][1] if suffix else '-',
        'Porter_Stem': stemmer.stem(w),
        'Lemma (verb)': lemmatizer.lemmatize(w, wordnet.VERB),
        'Lemma (noun)': lemmatizer.lemmatize(w, wordnet.NOUN),
    }


words = ['unhappiness', 'rewriting', 'misunderstood', 'players', 'quickly', 'replayed',
         'internationalization', 'enjoyable', 'fastest', 'understand', 'rest', 'reports']
print(pd.DataFrame([analyze_affixes(w) for w in words]))

# ---------------------------------------------------------------------------
# Step 4: Inflectional features with SpaCy's morphologizer
# token.morph gives Universal Dependencies features: Number=Plur, Tense=Past,
# Degree=Sup ... Handles irregular forms (children -> child).
# ---------------------------------------------------------------------------
sentence = 'The children were playing happily, but she wrote the longest essays and finished faster.'
doc = nlp(sentence)
morph_features = pd.DataFrame(
    [(t.text, t.lemma_, t.pos_, str(t.morph) if str(t.morph) else '-') for t in doc if not t.is_punct],
    columns=['Token', 'Lemma', 'POS', 'Morphological_Features']
)
print(morph_features)

# ---------------------------------------------------------------------------
# Step 5: POS tagging with NLTK (Penn Treebank tagset)
# ---------------------------------------------------------------------------
PENN_TAGS = {
    'CC': 'Coordinating conjunction', 'CD': 'Cardinal number', 'DT': 'Determiner',
    'IN': 'Preposition / subordinating conj.', 'JJ': 'Adjective', 'JJR': 'Adjective, comparative',
    'JJS': 'Adjective, superlative', 'MD': 'Modal verb', 'NN': 'Noun, singular',
    'NNS': 'Noun, plural', 'NNP': 'Proper noun, singular', 'PRP': 'Personal pronoun',
    'PRP$': 'Possessive pronoun', 'RB': 'Adverb', 'RBR': 'Adverb, comparative', 'TO': 'to',
    'VB': 'Verb, base form', 'VBD': 'Verb, past tense', 'VBG': 'Verb, gerund / present participle',
    'VBN': 'Verb, past participle', 'VBP': 'Verb, non-3rd person present',
    'VBZ': 'Verb, 3rd person singular present',
    'WDT': 'Wh-determiner', '.': 'Sentence-final punctuation', ',': 'Comma',
}

text = ('The quick brown fox jumps over the lazy dog. '
        'Students in Pune are learning natural language processing quickly.')
tokens = word_tokenize(text)
penn_tags = nltk.pos_tag(tokens)
penn_df = pd.DataFrame(penn_tags, columns=['Token', 'Penn_Tag'])
penn_df['Description'] = penn_df['Penn_Tag'].map(PENN_TAGS).fillna('Other')
print(penn_df)

# Step 6: Map Penn -> Universal tagset, and compare with SpaCy
spacy_doc = nlp(text)
compare_df = pd.DataFrame({
    'Token': [t for t, _ in penn_tags],
    'NLTK_Penn': [p for _, p in penn_tags],
    'NLTK_Universal': [map_tag('en-ptb', 'universal', p) for _, p in penn_tags],
    'SpaCy_Fine_Tag': [t.tag_ for t in spacy_doc],   # Penn-style
    'SpaCy_Universal': [t.pos_ for t in spacy_doc],  # UPOS
})
compare_df['Penn_Tags_Agree'] = compare_df['NLTK_Penn'] == compare_df['SpaCy_Fine_Tag']
print(f"NLTK and SpaCy agree on {compare_df['Penn_Tags_Agree'].sum()} of {len(compare_df)} Penn Treebank tags")
print(compare_df)

# ---------------------------------------------------------------------------
# Step 7: POS ambiguity - same word, different tags depending on context
# (target word, sentence, correct Penn tag for each occurrence of the word)
# ---------------------------------------------------------------------------
ambiguous_sentences = [
    ('can', 'I can can a can.', ['MD', 'VB', 'NN']),
    ('book', 'Please book a table and bring the book.', ['VB', 'NN']),
    ('flies', 'Time flies like an arrow, but fruit flies like a banana.', ['VBZ', 'NNS']),
    ('light', 'Turn on the light because the bag is light.', ['NN', 'JJ']),
]

rows = []
for word, sent, gold in ambiguous_sentences:
    nltk_tags = nltk.pos_tag(word_tokenize(sent))
    spacy_tags = [(t.text, t.tag_) for t in nlp(sent)]
    hits = [(nt, st) for (w, nt), (_, st) in zip(nltk_tags, spacy_tags) if w.lower() == word]
    for (nt, st), g in zip(hits, gold):
        rows.append({'Sentence': sent, 'Word': word, 'Correct_Tag': g,
                     'NLTK_Tag': nt, 'SpaCy_Tag': st, 'NLTK_OK': nt == g, 'SpaCy_OK': st == g})
amb_df = pd.DataFrame(rows)
print(f"NLTK correct: {amb_df['NLTK_OK'].sum()} / {len(amb_df)}   "
      f"SpaCy correct: {amb_df['SpaCy_OK'].sum()} / {len(amb_df)}")
print(amb_df)

# Step 8: POS tag distribution of a paragraph (Universal tags)
paragraph = ('Natural language processing is a branch of artificial intelligence. It helps computers '
             'understand, interpret and generate human language. Modern systems use large neural networks '
             'that are trained on huge collections of text. These models can translate languages, answer '
             'questions and summarize long documents quickly and accurately.')
para_tags = nltk.pos_tag(word_tokenize(paragraph), tagset='universal')
tag_counts = pd.Series([t for _, t in para_tags if t != '.']).value_counts()

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(tag_counts.index[::-1], tag_counts.values[::-1], color='#2a78d6', height=0.6)
for bar, v in zip(bars, tag_counts.values[::-1]):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2, str(v),
            va='center', fontsize=9, color='#333333')
ax.set_title('Universal POS tag distribution (punctuation excluded)', loc='left', fontsize=12, color='#333333')
ax.set_xlabel('Number of tokens', color='#333333')
ax.xaxis.grid(True, color='#e5e5e5'); ax.set_axisbelow(True)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Step 9: Train and evaluate taggers on the Penn Treebank (80/20 split)
# Backoff chain: Bigram -> Unigram -> Default. Each falls back when it has
# no answer for a word/context it never saw.
# ---------------------------------------------------------------------------
# Remove '-NONE-' trace tokens, which are not real words
tagged_sents = [[(w, t) for w, t in sent if t != '-NONE-'] for sent in treebank.tagged_sents()]
split = int(0.8 * len(tagged_sents))
train_sents, test_sents = tagged_sents[:split], tagged_sents[split:]
print(f'Training sentences: {len(train_sents)}, Test sentences: {len(test_sents)}')
print(f'Test tokens: {sum(len(s) for s in test_sents)}')

default_tagger = DefaultTagger('NN')
unigram_tagger = UnigramTagger(train_sents, backoff=default_tagger)
bigram_tagger = BigramTagger(train_sents, backoff=unigram_tagger)

random.seed(42)  # the perceptron shuffles training data; fixed seed -> reproducible
perceptron = PerceptronTagger(load=False)
perceptron.train(train_sents, nr_iter=5)

results = pd.DataFrame({
    'Tagger': ['Default (all NN)', 'Unigram + backoff', 'Bigram + backoff', 'Averaged Perceptron (trained here)'],
    'Accuracy_%': [round(100 * t.accuracy(test_sents), 2)
                   for t in (default_tagger, unigram_tagger, bigram_tagger, perceptron)],
})
print(results)  # expected ~15.5 / 87.0 / 87.7 / 95.4

fig, ax = plt.subplots(figsize=(8, 3.2))
bars = ax.barh(results['Tagger'][::-1], results['Accuracy_%'][::-1], color='#2a78d6', height=0.55)
for bar, v in zip(bars, results['Accuracy_%'][::-1]):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f'{v}%',
            va='center', fontsize=9, color='#333333')
ax.set_title('POS tagger accuracy on held-out Penn Treebank sentences', loc='left', fontsize=12, color='#333333')
ax.set_xlabel('Accuracy (%)', color='#333333')
ax.set_xlim(0, 110)
ax.xaxis.grid(True, color='#e5e5e5'); ax.set_axisbelow(True)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
plt.tight_layout()
plt.show()
