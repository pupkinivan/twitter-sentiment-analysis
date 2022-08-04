import pickle
import re

import numpy as np
from nltk import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


TFIDF_FILENAME = './resources/tfidf-vectorizer.pkl'


def filter_and_stem(input_string: str, stemmer: PorterStemmer) -> str:
    return stemmer.stem(re.sub('[^a-zA-Z]+', ' ', input_string).lower())


def load_vectorizer():
    return pickle.load(open(TFIDF_FILENAME, 'rb'))


def preprocess_sentence_for_inference(sentence: str, stemmer: PorterStemmer, vectorizer: TfidfVectorizer) -> np.ndarray:
    return vectorizer.transform([' '.join(list(map(lambda s: filter_and_stem(s, stemmer), sentence.split())))]).toarray()


def vectorize_corpus(corpus: list) -> np.ndarray:
    """Returns a tuple with two elements: a TF-IDF vectorizer instance and a vectorized text corpus."""
    vectorizer = TfidfVectorizer(max_features=1500, min_df=3, max_df=0.6)
    corpus_vectorized = vectorizer.fit_transform(corpus).toarray()
    return vectorizer, corpus_vectorized