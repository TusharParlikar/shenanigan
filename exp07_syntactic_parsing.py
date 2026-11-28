"""
Experiment 7: Syntactic analysis system - constituency parsing (CFG, PCFG) and
dependency parsing to discover grammatical relationships in complex sentences.

Install:
    pip install nltk spacy pandas
    python -m spacy download en_core_web_sm

CONCEPTS
- Parsing = finding the grammatical structure of a sentence.
- Constituency parsing groups words into nested phrases (NP, VP, PP) using a
  Context-Free Grammar (CFG): non-terminals, terminals (words), production
  rules like S -> NP VP, and a start symbol S. Chart parsers (CKY, Earley)
  store partial results to parse efficiently.
- Structural ambiguity: several valid trees. "I saw the man with the
  telescope" - the PP attaches to the verb (I used it) or the noun (he had it).
  #parses grows as Catalan numbers (1, 2, 5, 14) with each extra PP.
- PCFG: each rule has a probability; P(tree) = product of its rule
  probabilities; the Viterbi parser returns the most probable tree.
  From a treebank: P(A -> b) = Count(A -> b) / Count(A).
- Dependency parsing links each word to its HEAD with a labelled relation:
  nsubj (subject), dobj (direct object), det, amod, prep/pobj, relcl
  (relative clause), conj (conjunct). SpaCy uses a transition-based neural parser.

    Constituency                        | Dependency
    Nested phrases with phrase labels   | Word-to-word head-dependent relations
    Based on a CFG                      | Based on dependency grammar
    Grammar checking, phrase structure  | Relation extraction, free word order
"""

import nltk
import pandas as pd
import spacy
from spacy import displacy
from nltk import CFG, PCFG, ChartParser, ViterbiParser, induce_pcfg, Nonterminal
from nltk.corpus import treebank

for resource in ['punkt', 'punkt_tab', 'treebank']:
    nltk.download(resource, quiet=True)

pd.set_option('display.max_colwidth', None)
nlp = spacy.load('en_core_web_sm')

# Step 3: A Context-Free Grammar.
# NP -> NP PP and VP -> VP PP are what make PP-attachment ambiguous.
grammar = CFG.fromstring('''
    S     -> NP VP
    NP    -> Det N | Det Adj N | NP PP | Pron | PropN
    VP    -> V NP | VP PP | V
    PP    -> P NP
    Det   -> 'the' | 'a'
    Adj   -> 'detailed' | 'big'
    N     -> 'student' | 'report' | 'man' | 'telescope' | 'park' | 'library'
    Pron  -> 'I' | 'she'
    PropN -> 'Priya'
    V     -> 'wrote' | 'saw' | 'walked'
    P     -> 'with' | 'in'
''')
print('Start symbol:', grammar.start())
print('Number of production rules:', len(grammar.productions()))
print('\nPhrase-structure rules:')
for prod in grammar.productions():
    if not prod.is_lexical():  # lexical rules are the word rules (Det -> 'the')
        print('  ', prod)

# Step 4: Constituency parsing with a chart parser
parser = ChartParser(grammar)
sentence = 'the student wrote a detailed report in the library'.split()
trees = list(parser.parse(sentence))
print(f'Sentence: {" ".join(sentence)}')
print(f'Number of parse trees: {len(trees)}\n')

tree = trees[0]
tree.pretty_print()
print('Bracketed form:', tree)
print('Tree height:', tree.height())
print('Leaves (words):', tree.leaves())
print('\nNoun phrases found:')
for subtree in tree.subtrees(lambda t: t.label() == 'NP'):
    print('  ', ' '.join(subtree.leaves()))
print('\nRules used in this parse:')
for prod in tree.productions():
    print('  ', prod)

# Step 5: Structural ambiguity (PP attachment)
ambiguous = 'I saw the man with the telescope'.split()
amb_trees = list(parser.parse(ambiguous))
print(f'Sentence: {" ".join(ambiguous)}')
print(f'Number of parse trees: {len(amb_trees)}')
for i, t in enumerate(amb_trees, 1):
    pp = next(t.subtrees(lambda s: s.label() == 'PP'))
    parent = next(s for s in t.subtrees() if pp in list(s))  # node the PP hangs from
    print(f'\nParse {i}: "with the telescope" attaches to {parent.label()}')
    t.pretty_print()

# The number of parses grows rapidly (Catalan numbers) with each extra PP
growth = []
for s in ['I saw the man',
          'I saw the man with the telescope',
          'I saw the man in the park with the telescope',
          'I saw the man in the park with the telescope in the library']:
    growth.append({'Sentence': s, 'PPs': s.count(' with ') + s.count(' in '),
                   'Parse_Trees': len(list(parser.parse(s.split())))})
print(pd.DataFrame(growth))

# Step 6: A grammar also REJECTS ungrammatical word orders (0 parses)
for s in ['she walked in the park', 'walked the she park in']:
    n = len(list(parser.parse(s.split())))
    print(f'{s!r:30} -> {n} parse(s)', '(grammatical)' if n else '(rejected: no valid parse)')

# Step 7: Resolving ambiguity with a PCFG + Viterbi parser.
# Probabilities for each left-hand side must sum to 1.
pcfg = PCFG.fromstring('''
    S     -> NP VP        [1.0]
    NP    -> Det N        [0.4]
    NP    -> Det Adj N    [0.1]
    NP    -> NP PP        [0.2]
    NP    -> Pron         [0.2]
    NP    -> PropN        [0.1]
    VP    -> V NP         [0.6]
    VP    -> VP PP        [0.3]
    VP    -> V            [0.1]
    PP    -> P NP         [1.0]
    Det   -> 'the' [0.7] | 'a' [0.3]
    Adj   -> 'detailed' [0.5] | 'big' [0.5]
    N     -> 'student' [0.2] | 'report' [0.2] | 'man' [0.2] | 'telescope' [0.2] | 'park' [0.1] | 'library' [0.1]
    Pron  -> 'I' [0.5] | 'she' [0.5]
    PropN -> 'Priya' [1.0]
    V     -> 'wrote' [0.3] | 'saw' [0.4] | 'walked' [0.3]
    P     -> 'with' [0.5] | 'in' [0.5]
''')

# Probability of a parse tree = product of its rule probabilities
prob = {p.lhs(): {} for p in pcfg.productions()}
for p in pcfg.productions():
    prob[p.lhs()][p.rhs()] = p.prob()


def tree_prob(t):
    result = 1.0
    for p in t.productions():
        result *= prob[p.lhs()][p.rhs()]
    return result


for i, t in enumerate(amb_trees, 1):
    pp = next(t.subtrees(lambda s: s.label() == 'PP'))
    parent = next(s for s in t.subtrees() if pp in list(s))
    print(f'Parse {i} (PP attached to {parent.label()}): probability = {tree_prob(t):.3e}')

viterbi = ViterbiParser(pcfg)  # dynamic programming -> best tree only
best = next(viterbi.parse(ambiguous))
print(f'\nViterbi best parse (probability = {best.prob():.3e}):')
best.pretty_print()

# Step 8: Learning a PCFG from the Penn Treebank (rule counts -> probabilities).
# Trees are binarized into Chomsky Normal Form, creating symbols like NP|<DT-NN>.
productions = []
for t in treebank.parsed_sents():
    t.collapse_unary(collapsePOS=False)
    t.chomsky_normal_form(horzMarkov=2)
    productions += t.productions()

treebank_pcfg = induce_pcfg(Nonterminal('S'), productions)
print('Parsed sentences used:', len(treebank.parsed_sents()))
print('Production rules learned:', len(treebank_pcfg.productions()))

rows = []
for lhs in ('NP', 'VP'):
    rules = sorted(treebank_pcfg.productions(lhs=Nonterminal(lhs)), key=lambda p: -p.prob())[:5]
    rows += [{'LHS': lhs, 'Rule': f"{lhs} -> {' '.join(str(s) for s in p.rhs())}",
              'Probability': round(p.prob(), 4)} for p in rules]
print(pd.DataFrame(rows))

# Step 9: Dependency parsing of a complex sentence (relative clause + coordination)
complex_sentence = ('The researchers who developed the new model presented their results at the conference, '
                    'and the audience asked several difficult questions.')
doc = nlp(complex_sentence)
dep_df = pd.DataFrame([{
    'Token': t.text,
    'POS': t.pos_,
    'Dependency': t.dep_,
    'Meaning': spacy.explain(t.dep_) or '-',
    'Head': t.head.text,
    'Children': ', '.join(c.text for c in t.children) or '-',
} for t in doc])
print(dep_df)

# Step 10: Visualize the dependency tree (notebook: jupyter=True)
with open('exp07_dependency_tree.html', 'w', encoding='utf-8') as f:
    f.write(displacy.render(doc, style='dep', page=True, options={'compact': True, 'distance': 95}))

# Step 11: Extracting grammatical relations
CLAUSE_DEPS = ('ROOT', 'conj', 'relcl', 'advcl', 'ccomp', 'xcomp')


def phrase_without(token, exclude_deps=('relcl',)):
    # Text of a token's subtree, leaving out embedded clauses and punctuation
    skip = {t for child in token.children if child.dep_ in exclude_deps for t in child.subtree}
    return ' '.join(t.text for t in token.subtree if t not in skip and not t.is_punct)


def extract_svo(doc):
    # Subject-Verb-Object triples: find each subject, go to its head verb,
    # collect that verb's objects (direct, attribute, dative, prepositional).
    triples = []
    for token in doc:
        if token.dep_ in ('nsubj', 'nsubjpass'):
            verb = token.head
            objects = [(phrase_without(c), c.dep_) for c in verb.children if c.dep_ in ('dobj', 'attr', 'dative')]
            objects += [(f'{c.text} {phrase_without(g)}', f'prep ({c.text})')
                        for c in verb.children if c.dep_ == 'prep'
                        for g in c.children if g.dep_ == 'pobj']
            for obj_text, relation in objects:
                triples.append({'Subject': phrase_without(token), 'Verb': verb.lemma_,
                                'Object': obj_text, 'Relation': relation})
    return pd.DataFrame(triples)


print(extract_svo(doc))

print('Noun phrases (noun chunks):')
for chunk in doc.noun_chunks:
    print(f'   {chunk.text:30} root: {chunk.root.text:12} relation: {chunk.root.dep_} -> {chunk.root.head.text}')

print('\nClauses:')
for verb in doc:
    if verb.dep_ in CLAUSE_DEPS and verb.pos_ in ('VERB', 'AUX'):
        clause = phrase_without(verb, exclude_deps=('conj', 'cc', 'relcl', 'advcl', 'ccomp'))
        print(f'   [{verb.dep_:5}] {clause}')

# Step 12: Constituency vs dependency on the same sentence.
# Note: CFG attaches "in the library" to the verb, SpaCy attaches it to "report".
same = 'the student wrote a detailed report in the library'
print('Constituency tree (NLTK CFG):')
next(parser.parse(same.split())).pretty_print()

same_doc = nlp(same)
print('Dependency relations (SpaCy):')
for t in same_doc:
    print(f'   {t.head.text:8} --{t.dep_:>6}--> {t.text}')
