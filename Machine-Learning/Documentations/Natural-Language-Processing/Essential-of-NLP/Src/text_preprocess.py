import sys
from pathlib import Path
config_path = Path(' /Machine-Learning/Documentations/Natural-Language-Processing/Essential-of-NLP')
sys.path.append(str(config_path))

import unicodedata
from config import EN_NLP

# Normalization functions
# Fungsi untuk menghitung panjang pesan
def message_length(x):
    return len(x)

# Fungsi untuk menghitung jumlah huruf kapital
def num_capitals(x):
    return sum(1 for char in x if char.isupper())

# Fungsi untuk menghitung jumlah tanda baca
def num_punctuation(x):
    return sum(1 for char in x if unicodedata.category(char).startswith('P'))

# Fungsi untuk menghitung jumlah kata per dokumen
def word_counts_v1(x, pipeline=EN_NLP):
    doc = pipeline(x)
    return sum([len(sentence) for sentence in doc.sents])

# Fungsi untuk menghitung jumlah kata per dokumen [update dengan stopword]
def word_counts_v2(text, pipeline=None, stopwords=None):
    doc = pipeline(text)
    return sum(1 for token in doc if token.text.lower() not in stopwords)

def word_counts_no_punct(text, pipeline, stopwords):
    doc = pipeline(text)
    
    # Token-level filtering to improve efficiency
    tokens = [token for token in doc if not token.is_space]
    total_tokens = len(tokens)
    
    word_count = sum(1 for token in tokens if token.text.lower() not in stopwords and token.pos_ not in ['PUNCT', 'SYM'])
    
    non_word_proportion = sum(1 for token in tokens if token.pos_ in ['PUNCT', 'SYM']) / total_tokens
    
    return word_count, non_word_proportion