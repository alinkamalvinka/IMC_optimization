import pandas as pd
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from spacy.lang.pt import Portuguese
from nltk.corpus import wordnet as wn

nltk.download('wordnet')

nltk.download('stopwords')
port_stop = set(nltk.corpus.stopwords.words('portuguese'))
newStopWords = ['produto','comprei', 'produtos']
port_stop.update(newStopWords)


def pre_process(data, column_name):

    raw_corpora = data.copy()
    raw_corpora[column_name] = raw_corpora[column_name].fillna(" ")

    # No hyphens
    raw_corpora[column_name] = raw_corpora[column_name].map(lambda x: re.sub('—', ' ', x))

    # No hyphens2
    raw_corpora[column_name] = raw_corpora[column_name].map(lambda x: re.sub('-', ' ', x))

    # Substitute punctuation with space  no_hyphens = re.sub('-','','123-45-6789')
    raw_corpora[column_name] = raw_corpora[column_name].map(lambda x: re.sub('[,.!?]', ' ', x))

    # Remove punctuation
    raw_corpora[column_name] = raw_corpora[column_name].map(lambda x: re.sub('[^a-zA-Z ]+', ' ', x))

    # Remove odd letters
    raw_corpora[column_name] = raw_corpora[column_name].map(lambda x: re.sub('ó', 'o', x))

   # Convert the titles to lowercase
    raw_corpora[column_name] = raw_corpora[column_name].map(lambda x: x.lower())

    return raw_corpora


def tokenize(text):
    parser = Portuguese()
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in port_stop]
    tokens = [get_lemma(token) for token in tokens]
    tokens = [get_lemma2(token) for token in tokens]
    return tokens


def lda_preprocess_pipeline(data_for_lda, text_column_name):
    pre_processed_docs = pre_process(data_for_lda, text_column_name)
    processed_docs = pre_processed_docs[text_column_name].map(prepare_text_for_lda)
    return processed_docs

