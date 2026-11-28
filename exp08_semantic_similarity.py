"""
Experiment 8: Semantic similarity tool using WordNet and pre-trained word
embeddings (GloVe) for words, phrases and documents.

Install:
    pip install nltk gensim spacy scipy pandas numpy matplotlib
    python -m spacy download en_core_web_md

CONCEPTS
- Semantic similarity = closeness in MEANING (car/automobile high, car/banana
  low). Differs from RELATEDNESS: coffee and cup are related, not similar.
- WordNet: words grouped into synsets (one per meaning), linked by
  hypernymy (dog -> canine -> animal), meronymy (wheel part-of car),
  antonymy (hot / cold). Similarity from positions in the noun hierarchy:
    Path        = 1 / (1 + shortest path length)
    Wu-Palmer   = 2 * depth(LCS) / (depth(s1) + depth(s2)),
                  LCS = Least Common Subsumer (most specific common ancestor)
    Leacock-Chodorow = -log(path / (2 * max depth))
    Lin         = uses information content of the LCS (corpus frequencies)
- Word embeddings (GloVe) learn dense vectors from co-occurrence statistics
  (distributional hypothesis: similar contexts -> similar meaning).
    cosine(a, b) = a.b / (|a| |b|)
  They support analogies (king - man + woman ~ queen) but rate antonyms as
  similar, because opposites occur in the same contexts.
- Sentence/document similarity: average word vectors, or match words via
  WordNet; both ignore word order.
- Evaluation: Spearman rank correlation with human ratings (WordSim-353).
"""

import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import gensim.downloader as api
from itertools import product
from scipy.stats import spearmanr
from nltk.corpus import wordnet as wn, wordnet_ic, stopwords
from nltk.tokenize import word_tokenize
from gensim.test.utils import datapath

for resource in ['wordnet', 'omw-1.4', 'wordnet_ic', 'punkt', 'punkt_tab', 'stopwords']:
    nltk.download(resource, quiet=True)

pd.set_option('display.max_colwidth', None)

# Step 2: Load models
brown_ic = wordnet_ic.ic('ic-brown.dat')     # information content for Lin similarity
glove = api.load('glove-wiki-gigaword-100')  # pre-trained GloVe vectors (~128 MB)
nlp = spacy.load('en_core_web_md')           # SpaCy model with 300-d word vectors
stop_words = set(stopwords.words('english'))

print('WordNet synsets:', len(list(wn.all_synsets())))
print('GloVe vocabulary:', len(glove.key_to_index), '| vector size:', glove.vector_size)
print('SpaCy vectors:', nlp.vocab.vectors.shape)

# Step 3: Exploring WordNet - polysemy ("bank" has many senses)
print("Senses of the word 'bank':")
print(pd.DataFrame([(s.name(), s.pos(), s.definition()) for s in wn.synsets('bank')[:6]],
                   columns=['Synset', 'POS', 'Definition']))

dog = wn.synset('dog.n.01')
print('Synonyms (lemmas) of dog.n.01:', dog.lemma_names())
print('\nHypernym path (most general -> most specific):')
print('   ' + ' -> '.join(s.name().split('.')[0] for s in dog.hypernym_paths()[0]))
print('\nHyponyms (more specific kinds):', [s.name().split('.')[0] for s in dog.hyponyms()[:8]])
print('Part meronyms of car.n.01:', [s.name().split('.')[0] for s in wn.synset('car.n.01').part_meronyms()[:8]])
print('Antonym of good (adjective):', wn.synset('good.a.01').lemmas()[0].antonyms()[0].name())

lcs = dog.lowest_common_hypernyms(wn.synset('cat.n.01'))[0]
print('\nLeast Common Subsumer of dog and cat:', lcs.name(), '| depth:', lcs.min_depth())

# Step 4: WordNet similarity measures
word_pairs = [('car', 'automobile'), ('coast', 'shore'), ('dog', 'wolf'), ('dog', 'cat'),
              ('coffee', 'tea'), ('car', 'bicycle'), ('doctor', 'nurse'), ('coffee', 'cup'),
              ('king', 'cabbage'), ('computer', 'banana')]


def wordnet_scores(w1, w2):
    # Words have several senses: take the most similar pair of noun senses
    s1, s2 = wn.synsets(w1, pos=wn.NOUN), wn.synsets(w2, pos=wn.NOUN)
    if not s1 or not s2:
        return dict.fromkeys(['Path', 'Wu-Palmer', 'Leacock-Chodorow', 'Lin'], np.nan)
    pairs = list(product(s1, s2))
    return {
        'Path': max(a.path_similarity(b) or 0 for a, b in pairs),
        'Wu-Palmer': max(a.wup_similarity(b) or 0 for a, b in pairs),
        'Leacock-Chodorow': max(a.lch_similarity(b) for a, b in pairs),
        'Lin': max(a.lin_similarity(b, brown_ic) for a, b in pairs),
    }


wn_df = pd.DataFrame([{'Word 1': a, 'Word 2': b, **wordnet_scores(a, b)} for a, b in word_pairs]).round(3)
print(wn_df)

# Step 5: GloVe cosine similarity
emb_df = pd.DataFrame([{'Word 1': a, 'Word 2': b, 'GloVe_Cosine': round(float(glove.similarity(a, b)), 3)}
                       for a, b in word_pairs])
print(emb_df)

# Nearest neighbours in vector space (note "python" -> monty: corpus sense wins)
for word in ['computer', 'pune', 'python']:
    neighbours = ', '.join(f'{w} ({s:.2f})' for w, s in glove.most_similar(word, topn=6))
    print(f'{word:9} -> {neighbours}')

# Why SpaCy's md model is NOT used for single words: its pruned table maps
# many words onto 20k shared vectors, so dog and cat get the SAME vector.
vectors = nlp.vocab.vectors
print(f'SpaCy en_core_web_md: {vectors.n_keys} words share {vectors.shape[0]} unique vectors')
for w1, w2 in [('dog', 'cat'), ('car', 'automobile')]:
    same = np.array_equal(nlp.vocab[w1].vector, nlp.vocab[w2].vector)
    print(f'   {w1}/{w2}: identical vectors = {same}, '
          f'SpaCy similarity = {nlp.vocab[w1].similarity(nlp.vocab[w2]):.3f}, '
          f'GloVe similarity = {glove.similarity(w1, w2):.3f}')

# Step 6: Word analogies - a : b :: c : ?  solved as  b - a + c
analogies = [('man', 'king', 'woman'), ('france', 'paris', 'india'), ('walk', 'walked', 'swim'),
             ('good', 'better', 'bad'), ('india', 'rupee', 'japan')]
rows = []
for a, b, c in analogies:
    answer, score = glove.most_similar(positive=[b, c], negative=[a], topn=1)[0]
    rows.append({'Analogy': f'{a} : {b} :: {c} : ?', 'Vector arithmetic': f'{b} - {a} + {c}',
                 'Prediction': answer, 'Cosine': round(score, 3)})
print(pd.DataFrame(rows))

# Step 7: WordNet vs embeddings
compare = wn_df[['Word 1', 'Word 2', 'Wu-Palmer', 'Lin']].merge(emb_df, on=['Word 1', 'Word 2'])
print('Spearman correlation between Wu-Palmer and GloVe on these pairs:',
      round(spearmanr(compare['Wu-Palmer'], compare['GloVe_Cosine'])[0], 3))
print(compare)

# Antonyms: GloVe scores them HIGH (relatedness), WordNet knows they are opposites
antonyms = [('hot', 'cold'), ('good', 'bad'), ('increase', 'decrease'), ('happy', 'sad')]
print(pd.DataFrame([{'Pair': f'{a} / {b}', 'GloVe_Cosine': round(float(glove.similarity(a, b)), 3),
                     'WordNet_Antonyms': any(ant.name() == b for s in wn.synsets(a)
                                             for l in s.lemmas() for ant in l.antonyms())}
                    for a, b in antonyms]))


# Step 8: Phrase and sentence similarity
def content_words(text):
    return [w for w in word_tokenize(text.lower()) if w.isalpha() and w not in stop_words]


def glove_sentence_similarity(t1, t2):
    # Sentence vector = mean of its word vectors, then cosine
    v1 = np.mean([glove[w] for w in content_words(t1) if w in glove], axis=0)
    v2 = np.mean([glove[w] for w in content_words(t2) if w in glove], axis=0)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def best_wordnet_match(word, others):
    synsets = wn.synsets(word)
    scores = [a.wup_similarity(b) or 0 for o in others for a, b in product(synsets, wn.synsets(o))]
    return max(scores, default=0)


def wordnet_sentence_similarity(t1, t2):
    # Each word is matched to its best WordNet partner in the other sentence,
    # averaged in both directions (symmetric).
    w1, w2 = content_words(t1), content_words(t2)
    forward = np.mean([best_wordnet_match(w, w2) for w in w1])
    backward = np.mean([best_wordnet_match(w, w1) for w in w2])
    return float((forward + backward) / 2)


sentence_pairs = [
    ('A man is playing a guitar.', 'A person is playing a musical instrument.'),
    ('The cat sat on the mat.', 'A kitten was resting on the rug.'),
    ('The stock market fell sharply today.', 'Share prices dropped steeply this morning.'),
    ('The stock market fell sharply today.', 'The chef cooked a delicious pasta dinner.'),
    ('The movie was good.', 'The movie was not good.'),  # negation lost ("not" is a stop word)
    ('The dog bit the man.', 'The man bit the dog.'),    # word order ignored
]
sent_df = pd.DataFrame([{
    'Sentence 1': a, 'Sentence 2': b,
    'WordNet': round(wordnet_sentence_similarity(a, b), 3),
    'GloVe_Avg': round(glove_sentence_similarity(a, b), 3),
    'SpaCy': round(nlp(a).similarity(nlp(b)), 3),
} for a, b in sentence_pairs])
print(sent_df)

# Step 9: Document similarity matrix
documents = {
    'Cricket': 'India won the cricket match after the batsman scored a century in the final over.',
    'Football': 'The football team scored two goals and won the championship trophy.',
    'Economy': 'The central bank raised interest rates to control inflation in the economy.',
    'Markets': 'Stock prices fell as investors worried about rising inflation and interest rates.',
    'Cooking': 'Add chopped onions and tomatoes to the pan and cook the curry slowly.',
}
names = list(documents)
sim_matrix = np.array([[glove_sentence_similarity(documents[a], documents[b]) for b in names] for a in names])

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(sim_matrix, cmap='Blues', vmin=sim_matrix.min(), vmax=1)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha='right')
ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
for i in range(len(names)):
    for j in range(len(names)):
        val = sim_matrix[i, j]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=10,
                color='white' if val > (sim_matrix.min() + 1) / 2 else '#333333')
ax.set_title('Document similarity (cosine of averaged GloVe vectors)', loc='left', fontsize=11, color='#333333')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# argsort()[-2] -> second highest (the highest is the document itself)
most_similar = {n: names[np.argsort(sim_matrix[i])[-2]] for i, n in enumerate(names)}
print('Most similar document to each document:', most_similar)

# Step 10: Evaluation against human judgments (WordSim-353, ships with Gensim)
ws = pd.read_csv(datapath('wordsim353.tsv'), sep='\t', comment='#', header=None, names=['w1', 'w2', 'human'])
ws['w1'], ws['w2'] = ws['w1'].str.lower(), ws['w2'].str.lower()
print('Word pairs:', len(ws))


def best_wup(w1, w2):
    scores = [a.wup_similarity(b) for a, b in product(wn.synsets(w1), wn.synsets(w2))]
    scores = [s for s in scores if s is not None]
    return max(scores) if scores else np.nan


ws['GloVe'] = [float(glove.similarity(a, b)) if a in glove and b in glove else np.nan
               for a, b in zip(ws.w1, ws.w2)]
ws['WordNet_WUP'] = [best_wup(a, b) for a, b in zip(ws.w1, ws.w2)]

# Spearman's rho compares RANKINGS, so the measures need not be on the same scale
eval_rows = []
for col in ['WordNet_WUP', 'GloVe']:
    valid = ws.dropna(subset=[col])
    rho, p = spearmanr(valid['human'], valid[col])
    eval_rows.append({'Method': col, 'Pairs_Covered': len(valid), 'Spearman_rho': round(rho, 3),
                      'p_value': f'{p:.1e}'})
eval_df = pd.DataFrame(eval_rows)
print(eval_df)  # expected WordNet ~0.348, GloVe ~0.533

fig, ax = plt.subplots(figsize=(7, 2.6))
bars = ax.barh(eval_df['Method'][::-1], eval_df['Spearman_rho'][::-1], color='#2a78d6', height=0.5)
for bar, v in zip(bars, eval_df['Spearman_rho'][::-1]):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f'{v:.3f}',
            va='center', fontsize=9, color='#333333')
ax.set_xlim(0, 1)
ax.set_title("Agreement with human ratings on WordSim-353 (Spearman's rho)", loc='left', fontsize=11, color='#333333')
ax.set_xlabel("Spearman's rho", color='#333333')
ax.xaxis.grid(True, color='#e5e5e5'); ax.set_axisbelow(True)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
plt.tight_layout()
plt.show()

# Where the methods disagree: WordNet wins on synonyms, GloVe on topical relatedness
ws['Human_scaled'] = ws['human'] / 10
both = ws.dropna(subset=['GloVe', 'WordNet_WUP']).copy()
both['Gap'] = both['WordNet_WUP'] - both['GloVe']
disagree = pd.concat([both.nlargest(4, 'Gap'), both.nsmallest(4, 'Gap')])
print(disagree[['w1', 'w2', 'Human_scaled', 'WordNet_WUP', 'GloVe']].round(3).reset_index(drop=True))

# Google analogy test set (~19,500 questions), restricted to top-30k words
score, sections = glove.evaluate_word_analogies(datapath('questions-words.txt'), restrict_vocab=30000)
print(f'Overall analogy accuracy: {score:.1%}')
print(pd.DataFrame([{'Section': s['section'],
                     'Questions': len(s['correct']) + len(s['incorrect']),
                     'Accuracy_%': round(100 * len(s['correct']) / max(1, len(s['correct']) + len(s['incorrect'])), 1)}
                    for s in sections if s['section'] != 'Total accuracy']))
