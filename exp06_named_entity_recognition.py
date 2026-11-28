"""
Experiment 6: Named Entity Recognition (NER) application with SpaCy to identify
persons, organizations, locations, dates and monetary values.

Install:
    pip install spacy pandas matplotlib
    python -m spacy download en_core_web_sm

CONCEPTS
- NER locates named entities in text and classifies them:
  "Priya Sharma joined Infosys in Pune in 2025" ->
  Priya Sharma (PERSON), Infosys (ORG), Pune (GPE), 2025 (DATE).
- SpaCy English models know 18 OntoNotes types: PERSON, ORG, GPE (countries,
  cities), LOC (other locations), FAC (buildings), DATE, TIME, MONEY,
  PERCENT, CARDINAL, ...
- IOB tagging: NER is sequence labelling; each token is B (beginning),
  I (inside) or O (outside) an entity: Priya/B-PERSON Sharma/I-PERSON joined/O.
- Approaches: rule-based, statistical (CRF), neural. SpaCy's statistical
  model can be combined with the rule-based EntityRuler for domain names.
- Evaluation: a predicted entity is correct only if text AND label match.
      Precision = TP / (TP + FP)
      Recall    = TP / (TP + FN)
      F1        = 2PR / (P + R)
  Lenient evaluation also accepts overlapping text with the correct label,
  separating boundary errors from complete misses.
"""

import spacy
from spacy import displacy
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth', None)

# Step 2: Load the model and list its entity labels
nlp = spacy.load('en_core_web_sm')
print('SpaCy version:', spacy.__version__)
print('Pipeline components:', nlp.pipe_names)

ner_labels = nlp.get_pipe('ner').labels
print(f'\nThe model recognizes {len(ner_labels)} entity types:')
print(pd.DataFrame({'Label': ner_labels, 'Description': [spacy.explain(l) for l in ner_labels]}))

# Step 3: Input text
text = ('On 15 March 2026, Priya Sharma, the CEO of Nimbus Technologies, met officials from the '
        'Reserve Bank of India in Mumbai to discuss a $3 billion investment in cloud infrastructure. '
        'The agreement is expected to create 10,000 jobs in Maharashtra by 2030. Later that week, '
        'Rahul Mehta from Infosys presented the project at the Indian Institute of Science in Bengaluru, '
        'where the team reported a 12% improvement in processing speed. The first phase will cost about '
        'Rs 500 crore, and work begins next Monday.')
doc = nlp(text)
print(text)
print('\nNumber of entities found:', len(doc.ents))

# Step 4: Entities with labels and character offsets
entities_df = pd.DataFrame(
    [(ent.text, ent.label_, spacy.explain(ent.label_), ent.start_char, ent.end_char) for ent in doc.ents],
    columns=['Entity', 'Label', 'Description', 'Start_Char', 'End_Char']
)
print(entities_df)

# Step 5: Visualize with displaCy (in a notebook use jupyter=True; here save HTML)
with open('exp06_entities.html', 'w', encoding='utf-8') as f:
    f.write(displacy.render(doc, style='ent', page=True))
print('Entity visualization saved to exp06_entities.html')

# Step 6: Token-level IOB tags for the first sentence
first_sentence = list(doc.sents)[0]
iob_df = pd.DataFrame(
    [(t.text, t.ent_iob_, t.ent_type_ or '-', f'{t.ent_iob_}-{t.ent_type_}' if t.ent_type_ else 'O')
     for t in first_sentence],
    columns=['Token', 'IOB', 'Entity_Type', 'Combined_Tag']
)
print(iob_df)

# Step 7: Entity statistics and grouping by category
label_counts = entities_df['Label'].value_counts()
fig, ax = plt.subplots(figsize=(8, 0.5 * len(label_counts) + 1))
bars = ax.barh(label_counts.index[::-1], label_counts.values[::-1], color='#2a78d6', height=0.6)
for bar, v in zip(bars, label_counts.values[::-1]):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2, str(v),
            va='center', fontsize=9, color='#333333')
ax.set_title('Entities found by type', loc='left', fontsize=12, color='#333333')
ax.set_xlabel('Number of entities', color='#333333')
ax.set_xlim(0, label_counts.max() * 1.15)
ax.xaxis.grid(True, color='#e5e5e5'); ax.set_axisbelow(True)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
plt.tight_layout()
plt.show()

CATEGORIES = {
    'Persons': ['PERSON'],
    'Organizations': ['ORG'],
    'Locations': ['GPE', 'LOC', 'FAC'],
    'Dates': ['DATE', 'TIME'],
    'Monetary values': ['MONEY'],
    'Numbers & percentages': ['CARDINAL', 'PERCENT', 'QUANTITY', 'ORDINAL'],
}
grouped = pd.DataFrame([
    {'Category': cat, 'Entities': ', '.join(e.text for e in doc.ents if e.label_ in labels) or '-'}
    for cat, labels in CATEGORIES.items()
])
print(grouped)

# ---------------------------------------------------------------------------
# Step 8: Extending NER with custom rules (EntityRuler)
# Added BEFORE 'ner' so rule matches take priority; statistical NER fills the rest.
# Patterns can be exact phrases or token-attribute lists (LOWER, POS, ...).
# ---------------------------------------------------------------------------
nlp_custom = spacy.load('en_core_web_sm')
ruler = nlp_custom.add_pipe('entity_ruler', before='ner')
ruler.add_patterns([
    {'label': 'ORG', 'pattern': 'Nimbus Technologies'},
    {'label': 'ORG', 'pattern': 'MIT ADT University'},
    {'label': 'TECH', 'pattern': [{'LOWER': 'natural'}, {'LOWER': 'language'}, {'LOWER': 'processing'}]},
    {'label': 'TECH', 'pattern': [{'LOWER': 'spacy'}]},
    {'label': 'TECH', 'pattern': [{'LOWER': 'python'}]},
])
print('Custom pipeline:', nlp_custom.pipe_names)

test_text = ('Students at MIT ADT University use Python and spaCy for natural language processing, '
             'and Nimbus Technologies offered internships to 25 of them in Pune.')
before = [(e.text, e.label_) for e in nlp(test_text).ents]
after = [(e.text, e.label_) for e in nlp_custom(test_text).ents]
print('\nBefore EntityRuler:', before)
print('After EntityRuler: ', after)

with open('exp06_custom_entities.html', 'w', encoding='utf-8') as f:
    f.write(displacy.render(nlp_custom(test_text), style='ent', page=True,
                            options={'colors': {'TECH': '#9be7c4'}}))

# ---------------------------------------------------------------------------
# Step 9: Evaluation on a manually annotated test set
# ---------------------------------------------------------------------------
gold_data = [
    ('Ananya Iyer joined Google in London in January 2025.',
     [('Ananya Iyer', 'PERSON'), ('Google', 'ORG'), ('London', 'GPE'), ('January 2025', 'DATE')]),
    ('Sahyadri Motors reported a 12% increase in revenue on Monday.',
     [('Sahyadri Motors', 'ORG'), ('12%', 'PERCENT'), ('Monday', 'DATE')]),
    ('Arjun Nair watched India play Australia at the Melbourne Cricket Ground last Sunday.',
     [('Arjun Nair', 'PERSON'), ('India', 'GPE'), ('Australia', 'GPE'),
      ('the Melbourne Cricket Ground', 'FAC'), ('last Sunday', 'DATE')]),
    ('The Reserve Bank of India kept the repo rate at 6.5 percent in August 2024.',
     [('The Reserve Bank of India', 'ORG'), ('6.5 percent', 'PERCENT'), ('August 2024', 'DATE')]),
    ('Meera Joshi paid $250 for a flight from Delhi to Paris on 3 May.',
     [('Meera Joshi', 'PERSON'), ('$250', 'MONEY'), ('Delhi', 'GPE'), ('Paris', 'GPE'), ('3 May', 'DATE')]),
    ('Infosys opened a new office in Pune with 2,000 employees.',
     [('Infosys', 'ORG'), ('Pune', 'GPE'), ('2,000', 'CARDINAL')]),
]


def count_matches(gold, pred, strict=True):
    # Strict: same text and label. Lenient: same label and overlapping text.
    unused = list(pred)
    tp = 0
    for g_text, g_label in gold:
        for p in unused:
            p_text, p_label = p
            same = (p_text == g_text) if strict else (p_text in g_text or g_text in p_text)
            if same and p_label == g_label:
                tp += 1
                unused.remove(p)
                break
    return tp, len(pred) - tp, len(gold) - tp  # TP, FP, FN


rows, eval_rows = [], []
for sentence, gold in gold_data:
    pred = [(e.text, e.label_) for e in nlp(sentence).ents]
    for mode in ('Strict', 'Lenient'):
        tp, fp, fn = count_matches(gold, pred, strict=(mode == 'Strict'))
        eval_rows.append({'Mode': mode, 'TP': tp, 'FP': fp, 'FN': fn})
    rows.append({'Sentence': sentence, 'Gold': gold, 'Predicted': pred,
                 'Missed (FN)': [x for x in gold if x not in pred] or '-',
                 'Wrong (FP)': [x for x in pred if x not in gold] or '-'})
print(pd.DataFrame(rows)[['Sentence', 'Missed (FN)', 'Wrong (FP)']])


def scores(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return round(precision, 3), round(recall, 3), round(f1, 3)


totals = pd.DataFrame(eval_rows).groupby('Mode', sort=False)[['TP', 'FP', 'FN']].sum()
totals[['Precision', 'Recall', 'F1']] = totals.apply(lambda r: pd.Series(scores(r.TP, r.FP, r.FN)), axis=1)
print(totals)  # expected strict F1 0.870, lenient F1 0.913

# Strict per-label scores (set operations: & = TP, p-g = FP, g-p = FN)
per_label = {}
for sentence, gold in gold_data:
    pred = {(e.text, e.label_) for e in nlp(sentence).ents}
    gold = set(gold)
    for label in {l for _, l in gold | pred}:
        g = {x for x in gold if x[1] == label}
        p = {x for x in pred if x[1] == label}
        c = per_label.setdefault(label, [0, 0, 0])
        c[0] += len(g & p); c[1] += len(p - g); c[2] += len(g - p)

label_df = pd.DataFrame([(l, *c, *scores(*c)) for l, c in sorted(per_label.items())],
                        columns=['Label', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1'])
print(label_df)
