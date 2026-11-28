"""
Experiment 9: Design and evaluate language models - N-Gram (MLE, Laplace,
Kneser-Ney) and a neural LSTM - for next-word prediction and text generation.

Install:
    pip install nltk numpy pandas tensorflow matplotlib    (GPU recommended)

CONCEPTS
- A language model assigns a probability to a word sequence (chain rule):
      P(w1..wn) = prod_i P(wi | w1..wi-1)
- N-gram models use the MARKOV ASSUMPTION: the next word depends only on the
  previous N-1 words. Estimated from counts (Maximum Likelihood):
      P(wi | wi-1) = C(wi-1, wi) / C(wi-1)
- Smoothing: most N-grams never appear in training data, so MLE gives them
  probability 0 (=> infinite perplexity).
  * Laplace (add-one): add 1 to every count - moves TOO much mass to unseen events.
  * Interpolated Kneser-Ney: subtract a fixed discount D from observed counts
    and give that mass to a lower-order model built on CONTINUATION counts
    (how many different contexts a word follows). One of the best N-gram methods.
- Neural LM: an LSTM reads word embeddings, its gates keep information over
  long contexts, and a softmax layer outputs P(word) for every word.
- Evaluation:
      Perplexity PP = exp( -1/N * sum log P(wi | history) )   (lower = better)
      Top-k accuracy: is the true next word among the k most probable?
- Generation: greedy decoding (argmax) tends to repeat itself; sampling with
  a temperature gives more varied text. Distinct-n measures diversity.
"""

import math
import random
from collections import Counter, defaultdict

import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from nltk.corpus import gutenberg

nltk.download('gutenberg', quiet=True)
nltk.download('punkt', quiet=True)

pd.set_option('display.max_colwidth', None)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
print('TensorFlow version:', tf.__version__)
print('GPU available:', bool(tf.config.list_physical_devices('GPU')))

# ---------------------------------------------------------------------------
# Step 3: Load, preprocess and split the corpus (80/10/10)
# ---------------------------------------------------------------------------
# Lowercase, keep alphabetic words, drop very short sentences
sentences = [[w.lower() for w in s if w.isalpha()] for s in gutenberg.sents('austen-emma.txt')]
sentences = [s for s in sentences if len(s) >= 3]

random.Random(SEED).shuffle(sentences)
n = len(sentences)
train_raw = sentences[:int(0.8 * n)]
val_raw = sentences[int(0.8 * n):int(0.9 * n)]
test_raw = sentences[int(0.9 * n):]

# Vocabulary from TRAINING data only; words seen once become <unk>, so the
# model learns how to handle unknown words in the test set.
word_counts = Counter(w for s in train_raw for w in s)
vocab = ['<pad>', '<s>', '</s>', '<unk>'] + sorted(w for w, c in word_counts.items() if c >= 2)
word2id = {w: i for i, w in enumerate(vocab)}
id2word = {i: w for w, i in word2id.items()}
V = len(vocab)
PAD, BOS, EOS, UNK = 0, 1, 2, 3  # padding, begin-of-sentence, end-of-sentence, unknown


def encode(sentence):
    return [word2id.get(w, UNK) for w in sentence]


train, val, test = [list(map(encode, d)) for d in (train_raw, val_raw, test_raw)]

stats = pd.DataFrame({
    'Split': ['Train', 'Validation', 'Test'],
    'Sentences': [len(train), len(val), len(test)],
    'Words': [sum(map(len, d)) for d in (train, val, test)],
    'Unknown_words_%': [round(100 * sum(w == UNK for s in d for w in s) / sum(map(len, d)), 2)
                        for d in (train, val, test)],
})
print('Vocabulary size (including special tokens):', V)
print(stats)


# ---------------------------------------------------------------------------
# Step 4: N-gram language models
# ---------------------------------------------------------------------------
class NGramLM:
    # N-gram language model with MLE, Laplace, or interpolated Kneser-Ney estimation.
    # prob_vector(history) returns P(w | history) for every word id in the vocabulary.

    def __init__(self, n, method='kneser-ney', discount=0.75):
        self.n, self.method, self.D = n, method, discount

    def fit(self, sentences):
        self.raw = defaultdict(Counter)   # raw counts of the highest order: raw[context][word]
        ngram_types = defaultdict(set)    # distinct N-grams of every order
        for s in sentences:
            padded = [BOS] * (self.n - 1) + s + [EOS]
            for i in range(self.n - 1, len(padded)):
                self.raw[tuple(padded[i - self.n + 1:i])][padded[i]] += 1
                for m in range(1, self.n + 1):
                    ngram_types[m].add(tuple(padded[i - m + 1:i + 1]))

        # Continuation counts for lower orders: number of DISTINCT words that
        # precede (context, w). "Francisco" is frequent but follows only "San",
        # so its continuation count is low -> low unigram KN probability.
        self.cont = {m: defaultdict(Counter) for m in range(1, self.n)}
        for m in range(1, self.n):
            for gram in ngram_types[m + 1]:
                self.cont[m][gram[1:-1]][gram[-1]] += 1
        return self

    def _kn(self, history, m):
        # Recursive interpolated Kneser-Ney:
        # P(w|ctx) = max(c - D, 0)/total + (D * types/total) * P_lower(w)
        if m == 1:
            counts = self.cont[1][()] if self.n > 1 else self.raw[()]
            total, types = sum(counts.values()), len(counts)
            vec = np.full(V, self.D * types / total / V)  # interpolate with uniform
            for w, c in counts.items():
                vec[w] += max(c - self.D, 0) / total
            return vec

        ctx = tuple(history[-(m - 1):])
        counts = self.raw.get(ctx) if m == self.n else self.cont[m].get(ctx)
        lower = self._kn(history, m - 1)
        if not counts:  # unseen context: back off entirely
            return lower
        total, types = sum(counts.values()), len(counts)
        vec = (self.D * types / total) * lower
        for w, c in counts.items():
            vec[w] += max(c - self.D, 0) / total
        return vec

    def prob_vector(self, history):
        history = [BOS] * (self.n - 1) + list(history)
        if self.method == 'kneser-ney':
            vec = self._kn(history, self.n)
        else:
            counts = self.raw.get(tuple(history[-(self.n - 1):]) if self.n > 1 else (), Counter())
            total = sum(counts.values())
            vec = np.zeros(V)
            if self.method == 'laplace':
                V_pred = V - 2  # <pad> and <s> are never predicted
                vec[:] = 1.0 / (total + V_pred)
                for w, c in counts.items():
                    vec[w] = (c + 1) / (total + V_pred)
            elif total:  # MLE: unseen words stay at exactly 0
                for w, c in counts.items():
                    vec[w] = c / total
        vec[PAD] = vec[BOS] = 0.0  # never predicted
        return vec / vec.sum() if self.method == 'kneser-ney' else vec


def evaluate_lm(predict_fn, sentences):
    # Returns perplexity, top-1 and top-5 accuracy, and number of zero-probability words
    log_probs, top1, top5, zeros = [], 0, 0, 0
    for s in sentences:
        target_seq = s + [EOS]
        for i, target in enumerate(target_seq):
            p = predict_fn(s[:i])
            prob = p[target]
            if prob <= 0:
                zeros += 1
            log_probs.append(math.log(prob) if prob > 0 else -math.inf)
            best = np.argpartition(-p, 5)[:5]  # 5 most probable word ids (unordered)
            top5 += target in best
            top1 += target == best[np.argmax(p[best])]
    n = len(log_probs)
    ppl = math.exp(-sum(log_probs) / n) if zeros == 0 else math.inf
    return {'Perplexity': round(ppl, 1), 'Top1_Acc_%': round(100 * top1 / n, 2),
            'Top5_Acc_%': round(100 * top5 / n, 2), 'Zero_Prob_Words': zeros, 'Test_Tokens': n}


ngram_models = {
    'Unigram (Kneser-Ney)': NGramLM(1, 'kneser-ney').fit(train),
    'Bigram (Kneser-Ney)': NGramLM(2, 'kneser-ney').fit(train),
    'Trigram (MLE, no smoothing)': NGramLM(3, 'mle').fit(train),
    'Trigram (Laplace)': NGramLM(3, 'laplace').fit(train),
    'Trigram (Kneser-Ney)': NGramLM(3, 'kneser-ney').fit(train),
}
print('Distinct trigram contexts in training data:', len(ngram_models['Trigram (Kneser-Ney)'].raw))

# Sanity check: every smoothed distribution sums to 1
for name, m in ngram_models.items():
    if 'MLE' not in name:
        assert abs(m.prob_vector([word2id['she'], word2id['was']]).sum() - 1) < 1e-9
print('All smoothed distributions sum to 1.')

# Step 5: Evaluate N-gram models on the test set
results = [{'Model': name, **evaluate_lm(model.prob_vector, test)} for name, model in ngram_models.items()]
ngram_results = pd.DataFrame(results)
print(ngram_results)

# ---------------------------------------------------------------------------
# Step 6: Training data for the neural LM: (last CONTEXT words) -> next word
# ---------------------------------------------------------------------------
CONTEXT = 20  # number of previous words the LSTM can see


def make_sequences(sentences):
    X, y = [], []
    for s in sentences:
        seq = [BOS] + s + [EOS]
        for i in range(1, len(seq)):
            ctx = seq[max(0, i - CONTEXT):i]
            X.append([PAD] * (CONTEXT - len(ctx)) + ctx)  # left-pad short contexts
            y.append(seq[i])
    return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32)


X_train, y_train = make_sequences(train)
X_val, y_val = make_sequences(val)
X_test, y_test = make_sequences(test)
print('Training examples:', X_train.shape, '| Validation:', X_val.shape, '| Test:', X_test.shape)
print('\nExample (context -> next word):')
for i in range(3):
    ctx = ' '.join(id2word[t] for t in X_train[i] if t != PAD)
    print(f'   [{ctx}] -> {id2word[y_train[i]]}')

# Step 7: LSTM language model
# Embedding (mask_zero ignores <pad>) -> LSTM -> Dropout (regularization) -> softmax over V
lstm_model = tf.keras.Sequential([
    tf.keras.Input(shape=(CONTEXT,)),
    tf.keras.layers.Embedding(V, 128, mask_zero=True),
    tf.keras.layers.LSTM(256),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(V, activation='softmax'),
])
# Cross-entropy loss = mean negative log-likelihood, so exp(loss) = perplexity
lstm_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy')
lstm_model.summary()

# Early stopping: stop when validation loss stops improving (prevents overfitting)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
history = lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val),
                         epochs=15, batch_size=256, callbacks=[early_stop], verbose=2)

train_ppl = np.exp(history.history['loss'])
val_ppl = np.exp(history.history['val_loss'])
epochs = np.arange(1, len(train_ppl) + 1)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(epochs, train_ppl, color='#2a78d6', linewidth=2, marker='o', label='Training perplexity')
ax.plot(epochs, val_ppl, color='#eb6834', linewidth=2, marker='s', linestyle='--', label='Validation perplexity')
best = int(np.argmin(val_ppl))
ax.annotate(f'best: {val_ppl[best]:.1f}', (epochs[best], val_ppl[best]), textcoords='offset points',
            xytext=(0, 12), ha='center', fontsize=9, color='#333333')
ax.set_title('LSTM language model learning curves', loc='left', fontsize=12, color='#333333')
ax.set_xlabel('Epoch', color='#333333'); ax.set_ylabel('Perplexity', color='#333333')
ax.set_xticks(epochs)
ax.grid(True, color='#e5e5e5'); ax.set_axisbelow(True)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False)
plt.tight_layout()
plt.show()

# Step 8: Compare all models.
# LSTM predictions for every test position, in the same order evaluate_lm visits them.
lstm_probs = lstm_model.predict(X_test, batch_size=1024, verbose=0).astype(np.float64)
lstm_probs[:, [PAD, BOS]] = 0
lstm_probs /= lstm_probs.sum(axis=1, keepdims=True)

position = iter(range(len(lstm_probs)))
lstm_result = evaluate_lm(lambda history: lstm_probs[next(position)], test)

comparison = pd.concat([ngram_results, pd.DataFrame([{'Model': 'LSTM (neural)', **lstm_result}])],
                       ignore_index=True)
print(comparison)

plot_df = comparison[np.isfinite(comparison['Perplexity'])]
fig, ax = plt.subplots(figsize=(8, 3.6))
bars = ax.barh(plot_df['Model'][::-1], plot_df['Perplexity'][::-1], color='#2a78d6', height=0.55)
for bar, v in zip(bars, plot_df['Perplexity'][::-1]):
    ax.text(bar.get_width() * 1.04, bar.get_y() + bar.get_height() / 2, f'{v:.1f}',
            va='center', fontsize=9, color='#333333')
ax.set_xscale('log')
ax.set_title('Test perplexity (lower is better, log scale)', loc='left', fontsize=12, color='#333333')
ax.set_xlabel('Perplexity', color='#333333')
ax.xaxis.grid(True, color='#e5e5e5', which='both'); ax.set_axisbelow(True)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
ax.set_xlim(plot_df['Perplexity'].min() / 1.5, plot_df['Perplexity'].max() * 2.5)
plt.tight_layout()
plt.show()
print('Unsmoothed MLE trigram is not plotted because its perplexity is infinite.')

# Step 9: Next-word prediction
trigram = ngram_models['Trigram (Kneser-Ney)']


def lstm_prob_vector(history):
    ctx = ([BOS] + list(history))[-CONTEXT:]
    x = np.array([[PAD] * (CONTEXT - len(ctx)) + ctx])
    p = lstm_model.predict(x, verbose=0)[0].astype(np.float64)
    p[[PAD, BOS]] = 0
    return p / p.sum()


def top_words(p, k=5):
    best = np.argsort(-p)[:k]
    return ', '.join(f'{id2word[i]} ({p[i]:.2f})' for i in best)


contexts = ['i do not', 'she was very', 'mr knightley', 'it would be', 'emma could not help']
rows = []
for c in contexts:
    hist = encode(c.split())
    rows.append({'Context': c, 'Trigram (Kneser-Ney) top-5': top_words(trigram.prob_vector(hist)),
                 'LSTM top-5': top_words(lstm_prob_vector(hist))})
print(pd.DataFrame(rows))


# Step 10: Text generation and assessment
def generate(prob_fn, seed_text, length=25, mode='sample', temperature=0.8, rng=None):
    words = encode(seed_text.split())
    for _ in range(length):
        p = prob_fn(words).copy()
        p[UNK] = 0  # do not generate <unk>
        if mode == 'greedy':
            nxt = int(np.argmax(p))
        else:
            # Temperature < 1 sharpens the distribution, > 1 flattens it
            logits = np.log(p + 1e-12) / temperature
            p = np.exp(logits - logits.max()); p /= p.sum()
            nxt = int(rng.choice(V, p=p))
        if nxt == EOS:
            break
        words.append(nxt)
    return ' '.join(id2word[w] for w in words)


def distinct_n(texts, n):
    # Diversity: unique n-grams / total n-grams in the generated texts
    grams = [tuple(t.split()[i:i + n]) for t in texts for i in range(len(t.split()) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


generators = {
    'Bigram (Kneser-Ney)': ngram_models['Bigram (Kneser-Ney)'].prob_vector,
    'Trigram (Kneser-Ney)': trigram.prob_vector,
    'LSTM (neural)': lstm_prob_vector,
}
seeds = ['emma was', 'she could not', 'mr elton', 'it was a']

gen_rows, samples = [], {}
for name, fn in generators.items():
    rng = np.random.default_rng(SEED)
    sampled = [generate(fn, s, mode='sample', rng=rng) for s in seeds]
    samples[name] = {'Greedy': generate(fn, seeds[0], mode='greedy'), 'Sampled': sampled[0]}
    gen_rows.append({'Model': name,
                     'Distinct-1': round(distinct_n(sampled, 1), 3),
                     'Distinct-2': round(distinct_n(sampled, 2), 3),
                     'Avg_Length': round(np.mean([len(t.split()) for t in sampled]), 1)})

for name, s in samples.items():
    print(f'{name}\n   Greedy : {s["Greedy"]}\n   Sampled: {s["Sampled"]}\n')
print(pd.DataFrame(gen_rows))
