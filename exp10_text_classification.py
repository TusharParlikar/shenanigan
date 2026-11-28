"""
Experiment 10: Text classification with Machine Learning (Naive Bayes, Linear
SVM, Logistic Regression) to categorize Reuters news into predefined topics.

Install:
    pip install nltk scikit-learn pandas numpy matplotlib

CONCEPTS
- Text classification assigns predefined labels to documents (spam, news
  topics, sentiment). Pipeline:
      preprocessing -> feature extraction -> training -> evaluation -> prediction
- Features:
  * Bag-of-Words: raw word counts.
  * TF-IDF: weighs words frequent in a document but rare in the corpus:
        TF-IDF(t, d) = TF(t, d) * log(N / DF(t))
    sublinear_tf uses 1 + log(TF) to damp very frequent terms.
- Classifiers:
  * Multinomial Naive Bayes: Bayes' theorem assuming words are independent
    given the class: P(c|d) ~ P(c) * prod P(wi|c), with additive smoothing
    alpha. Very fast, strong baseline.
  * Linear SVM: maximum-margin hyperplane between classes; C trades margin
    width vs training errors. Excellent on sparse high-dimensional text.
  * Logistic Regression: softmax class probabilities with L2 regularization
    (strength 1/C); weights are interpretable.
- Evaluation: precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = harmonic mean.
  MACRO-averaging weights every class equally, so on imbalanced data macro-F1
  exposes poor small-class performance that accuracy hides.
  Confusion matrix shows which classes get confused. k-fold cross-validation
  + grid search gives reliable estimates and tuned hyperparameters.
"""

import re
import time
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import reuters, stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

nltk.download('reuters', quiet=True)
nltk.download('stopwords', quiet=True)
pd.set_option('display.max_colwidth', None)

# Step 2: Load Reuters-21578, single-label docs from 6 topics, official train/test split
CATEGORIES = ['earn', 'acq', 'crude', 'trade', 'money-fx', 'interest']
CATEGORY_NAMES = {'earn': 'Earnings', 'acq': 'Acquisitions', 'crude': 'Crude oil',
                  'trade': 'Trade', 'money-fx': 'Foreign exchange', 'interest': 'Interest rates'}

docs = [f for f in reuters.fileids()
        if len(reuters.categories(f)) == 1 and reuters.categories(f)[0] in CATEGORIES]
train_ids = [f for f in docs if f.startswith('training/')]
test_ids = [f for f in docs if f.startswith('test/')]
X_train_raw = [reuters.raw(f) for f in train_ids]
y_train = [reuters.categories(f)[0] for f in train_ids]
X_test_raw = [reuters.raw(f) for f in test_ids]
y_test = [reuters.categories(f)[0] for f in test_ids]

print(f'Training documents: {len(X_train_raw)} | Test documents: {len(X_test_raw)}')
print('\nSample document (' + y_train[0] + '):\n' + X_train_raw[0][:400] + '...')

# Step 3: Class distribution - heavily imbalanced (earn ~53%)
dist = pd.DataFrame({
    'Category': CATEGORIES,
    'Description': [CATEGORY_NAMES[c] for c in CATEGORIES],
    'Train': [y_train.count(c) for c in CATEGORIES],
    'Test': [y_test.count(c) for c in CATEGORIES],
})
dist['Train_%'] = (100 * dist['Train'] / dist['Train'].sum()).round(1)
print(dist.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 3.6))
bars = ax.barh(dist['Description'][::-1], dist['Train'][::-1], color='#2a78d6', height=0.6)
for bar, v, pct in zip(bars, dist['Train'][::-1], dist['Train_%'][::-1]):
    ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height() / 2, f'{v} ({pct}%)',
            va='center', fontsize=9, color='#333333')
ax.set_title('Training documents per category', loc='left', fontsize=12, color='#333333')
ax.set_xlabel('Number of documents', color='#333333')
ax.set_xlim(0, dist['Train'].max() * 1.25)
ax.xaxis.grid(True, color='#e5e5e5'); ax.set_axisbelow(True)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
plt.tight_layout()
plt.show()

# Step 4: Preprocessing
stop_words = set(stopwords.words('english'))


def preprocess(text):
    text = text.lower()
    text = re.sub(r'&lt;|&gt;|&amp;', ' ', text)  # HTML entities in Reuters text
    text = re.sub(r'[^a-z\s]', ' ', text)         # remove numbers and punctuation
    return ' '.join(w for w in text.split() if w not in stop_words and len(w) > 2)


X_train = [preprocess(d) for d in X_train_raw]
X_test = [preprocess(d) for d in X_test_raw]
print('Original :', X_train_raw[1][:250].replace('\n', ' '))
print('\nProcessed:', X_train[1][:250])

# Step 5: Feature extraction. min_df=2 drops words seen in only one document.
bow = CountVectorizer(min_df=2)
tfidf = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True)
X_bow = bow.fit_transform(X_train)
X_tfidf = tfidf.fit_transform(X_train)
print('Bag-of-Words matrix :', X_bow.shape, f'| non-zero entries: {X_bow.nnz / np.prod(X_bow.shape):.3%}')
print('TF-IDF (1-2 grams)  :', X_tfidf.shape, f'| non-zero entries: {X_tfidf.nnz / np.prod(X_tfidf.shape):.3%}')

# Highest TF-IDF terms in one document = what makes it distinctive
doc_vec = X_tfidf[1].toarray().ravel()
top = doc_vec.argsort()[::-1][:8]
features = tfidf.get_feature_names_out()
print(f'\nTop TF-IDF terms in training document 1 ({y_train[1]}):')
print(', '.join(f'{features[i]} ({doc_vec[i]:.2f})' for i in top))


# Step 6: Train and evaluate every (features x classifier) combination.
# Pipeline fits the vectorizer on training data only -> no test-set leakage.
def make_vectorizer(kind):
    return CountVectorizer(min_df=2) if kind == 'BoW' else TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True)


classifiers = {
    'Multinomial Naive Bayes': lambda: MultinomialNB(),
    'Linear SVM': lambda: LinearSVC(max_iter=10000, random_state=42),
    'Logistic Regression': lambda: LogisticRegression(max_iter=2000),
}

rows, trained = [], {}
for features_kind in ['BoW', 'TF-IDF']:
    for name, make_clf in classifiers.items():
        pipe = Pipeline([('vec', make_vectorizer(features_kind)), ('clf', make_clf())])
        start = time.time()
        pipe.fit(X_train, y_train)
        train_time = time.time() - start
        pred = pipe.predict(X_test)
        p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average='macro', zero_division=0)
        rows.append({'Features': features_kind, 'Classifier': name,
                     'Accuracy': round(accuracy_score(y_test, pred), 4),
                     'Macro_Precision': round(p, 4), 'Macro_Recall': round(r, 4),
                     'Macro_F1': round(f1, 4), 'Train_Time_s': round(train_time, 2)})
        trained[(features_kind, name)] = pipe
results = pd.DataFrame(rows)
print(results)  # note NB + TF-IDF: 87.8% accuracy but only 0.64 macro-F1

fig, ax = plt.subplots(figsize=(8, 3.8))
labels = [f'{c} ({f})' for f, c in zip(results['Features'], results['Classifier'])]
order = results['Macro_F1'].argsort().values
bars = ax.barh([labels[i] for i in order], results['Macro_F1'].values[order], color='#2a78d6', height=0.6)
for bar, v in zip(bars, results['Macro_F1'].values[order]):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f'{v:.3f}',
            va='center', fontsize=9, color='#333333')
ax.set_xlim(0, 1.08)
ax.set_title('Macro F1-score on the test set (higher is better)', loc='left', fontsize=12, color='#333333')
ax.set_xlabel('Macro F1-score', color='#333333')
ax.xaxis.grid(True, color='#e5e5e5'); ax.set_axisbelow(True)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
plt.tight_layout()
plt.show()

# Step 7: 5-fold stratified CV (keeps class proportions in every fold)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []
for name, make_clf in classifiers.items():
    pipe = Pipeline([('vec', make_vectorizer('TF-IDF')), ('clf', make_clf())])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1_macro')
    cv_rows.append({'Classifier': name, 'CV_Macro_F1_Mean': round(scores.mean(), 4),
                    'CV_Macro_F1_Std': round(scores.std(), 4)})
print(pd.DataFrame(cv_rows))

# Grid search: try each hyperparameter value, keep the best by CV macro-F1.
# 'clf__alpha' = parameter 'alpha' of the pipeline step named 'clf'.
param_grids = {
    'Multinomial Naive Bayes': {'clf__alpha': [0.01, 0.05, 0.1, 0.5, 1.0]},
    'Linear SVM': {'clf__C': [0.1, 0.5, 1, 5, 10]},
    'Logistic Regression': {'clf__C': [1, 10, 50, 100, 500]},
}
tuned_rows, tuned = [], {}
for name, make_clf in classifiers.items():
    pipe = Pipeline([('vec', make_vectorizer('TF-IDF')), ('clf', make_clf())])
    search = GridSearchCV(pipe, param_grids[name], cv=cv, scoring='f1_macro', n_jobs=-1)
    search.fit(X_train, y_train)
    pred = search.predict(X_test)
    tuned[name] = search.best_estimator_
    tuned_rows.append({'Classifier': name, 'Best_Params': search.best_params_,
                       'Best_CV_Macro_F1': round(search.best_score_, 4),
                       'Test_Accuracy': round(accuracy_score(y_test, pred), 4),
                       'Test_Macro_F1': round(precision_recall_fscore_support(y_test, pred, average='macro')[2], 4)})
tuned_df = pd.DataFrame(tuned_rows)
print(tuned_df)

# Step 8: Best model chosen by CV score (NOT test score - avoids test-set overfitting)
best_name = tuned_df.loc[tuned_df['Best_CV_Macro_F1'].idxmax(), 'Classifier']
best_model = tuned[best_name]
best_pred = best_model.predict(X_test)
print(f'Best model (selected by cross-validation): {best_name}\n')
print(classification_report(y_test, best_pred, labels=CATEGORIES, digits=3))

cm = confusion_matrix(y_test, best_pred, labels=CATEGORIES)  # rows = true, cols = predicted
names = [CATEGORY_NAMES[c] for c in CATEGORIES]
fig, ax = plt.subplots(figsize=(7.5, 6))
im = ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=35, ha='right')
ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
for i in range(len(names)):
    for j in range(len(names)):
        share = cm[i, j] / cm[i].sum()
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=10,
                color='white' if share > 0.5 else '#333333')
ax.set_xlabel('Predicted category', color='#333333'); ax.set_ylabel('True category', color='#333333')
ax.set_title(f'Confusion matrix: {best_name}\n(cell colour = share of the true category)',
             loc='left', fontsize=11, color='#333333')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# Step 9: Most informative words per class = largest Logistic Regression weights
lr = tuned['Logistic Regression']
vocab = lr.named_steps['vec'].get_feature_names_out()
coefs = lr.named_steps['clf'].coef_
classes = lr.named_steps['clf'].classes_
print(pd.DataFrame({CATEGORY_NAMES[c]: [vocab[i] for i in coefs[k].argsort()[::-1][:10]]
                    for k, c in enumerate(classes)}))

# Step 10: Error analysis - errors are mostly between overlapping topics
errors = [(i, t, p) for i, (t, p) in enumerate(zip(y_test, best_pred)) if t != p]
print(f'Misclassified test documents: {len(errors)} of {len(y_test)} ({len(errors) / len(y_test):.1%})\n')

pair_counts = pd.Series([f'{CATEGORY_NAMES[t]} -> {CATEGORY_NAMES[p]}' for _, t, p in errors]).value_counts()
print('Most frequent confusions (true -> predicted):')
print(pair_counts.head(5).to_string())

print(pd.DataFrame([{'True': CATEGORY_NAMES[t], 'Predicted': CATEGORY_NAMES[p],
                     'Document start': ' '.join(X_test_raw[i].split())[:160] + '...'}
                    for i, t, p in errors[:6]]))
