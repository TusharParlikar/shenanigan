# shenanigan

NLP & Applications Lab — Assignments 1 to 10, one Python file per experiment.
Each file starts with a **CONCEPTS** block (theory) and install commands, and the code is commented step by step.
The full report with outputs is in [NLP_Lab_Assignments_1-10_Tushar_Parlikar_final.pdf](NLP_Lab_Assignments_1-10_Tushar_Parlikar_final.pdf).

Simpler Jupyter notebook versions of all 10 experiments are in [ipynb/](ipynb/).

## Index

| # | Experiment | Code | Key concepts | Libraries |
|---|------------|------|--------------|-----------|
| 1 | NLP environment setup and verification | [exp01_nlp_environment_setup.py](exp01_nlp_environment_setup.py) | Tokenization, stop words, POS tagging, NER, Word2Vec, Transformer pipeline | NLTK, SpaCy, Gensim, Transformers |
| 2 | Text preprocessing pipeline | [exp02_text_preprocessing.py](exp02_text_preprocessing.py) | Normalization, stop-word removal, Porter stemming, WordNet lemmatization, vocabulary reduction | NLTK |
| 3 | Regex extraction and validation | [exp03_regex_extraction_validation.py](exp03_regex_extraction_validation.py) | Emails, URLs, phones, dates, hashtags; `finditer` vs `fullmatch`; lookarounds; calendar check; masking with `sub` | re, datetime |
| 4 | Bag-of-Words and N-gram models | [exp04_bow_ngrams.py](exp04_bow_ngrams.py) | Document-term matrix, sparsity, unigrams/bigrams/trigrams, frequency distributions, Zipf's law | NLTK, Scikit-learn |
| 5 | Morphological analysis and POS tagging | [exp05_morphology_pos_tagging.py](exp05_morphology_pos_tagging.py) | Inflectional vs derivational morphology, Penn vs Universal tagsets, ambiguity, Default/Unigram/Bigram/Perceptron taggers | NLTK, SpaCy |
| 6 | Named Entity Recognition | [exp06_named_entity_recognition.py](exp06_named_entity_recognition.py) | Entity types, IOB tagging, EntityRuler, strict vs lenient precision/recall/F1 | SpaCy |
| 7 | Constituency and dependency parsing | [exp07_syntactic_parsing.py](exp07_syntactic_parsing.py) | CFG, chart parsing, PP-attachment ambiguity, PCFG + Viterbi, treebank PCFG, SVO extraction | NLTK, SpaCy |
| 8 | Semantic similarity | [exp08_semantic_similarity.py](exp08_semantic_similarity.py) | WordNet (Path, Wu-Palmer, LCH, Lin), GloVe cosine, analogies, sentence/document similarity, WordSim-353 | NLTK, Gensim, SpaCy |
| 9 | N-gram and neural language models | [exp09_language_models.py](exp09_language_models.py) | Markov assumption, MLE/Laplace/Kneser-Ney smoothing, LSTM LM, perplexity, top-k accuracy, greedy vs sampling | NLTK, TensorFlow |
| 10 | Text classification | [exp10_text_classification.py](exp10_text_classification.py) | BoW vs TF-IDF, Naive Bayes, Linear SVM, Logistic Regression, macro-F1, cross-validation, grid search, confusion matrix | Scikit-learn, NLTK |

## Setup

```bash
pip install nltk spacy gensim transformers torch tensorflow scikit-learn scipy pandas numpy matplotlib
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md   # Experiment 8 only
```

NLTK resources (corpora, taggers, WordNet) are downloaded automatically inside each script.

## Run

```bash
python exp03_regex_extraction_validation.py
```

Experiments 6 and 7 save their displaCy visualizations as `.html` files in the current folder.
