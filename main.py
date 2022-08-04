import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import torch

import preprocessing
from classifiers.dnn import train_dnn, DNN_FILENAME
from classifiers.knn import train_knn, KNN_FILENAME
from preprocessing import vectorize_corpus, filter_and_stem, preprocess_sentence_for_inference, TFIDF_FILENAME


def train(algorithm):
    df = pd.read_csv("resources/Restaurant_Reviews.tsv",
                     delimiter="\t", quoting=3)
    y = df.iloc[:, 1].values  # Label: purchased/did-not-purchase
    stemmer = PorterStemmer()
    corpus = []
    print("Cleaning text set...")
    for i, row in df.iterrows():
        review_clean = [filter_and_stem(word, stemmer) for word in row["Review"].split() if
                        not (word in set(stopwords.words('english')))]
        corpus.append(' '.join(review_clean))
    print("Done clearning text")
    print("Vectorizing the corpus...")
    tfidf_vectorizer, corpus_vectorized = vectorize_corpus(corpus)
    print("Done vectorizing")

    X_train, X_test, y_train, y_test = train_test_split(corpus_vectorized, y, test_size=.2, random_state=0)

    if algorithm == 'knn':
        model = train_knn(X_train, y_train)
        model_filename = KNN_FILENAME
    elif algorithm == 'dnn':
        model = train_dnn(X_train, y_train, epochs=500)
        model_filename = DNN_FILENAME

    y_test_hat = evaluate(X_test, model, algorithm)
    conf_mat = confusion_matrix(y_test, y_test_hat)
    accuracy = accuracy_score(y_test, y_test_hat)
    print(f"Confusion matrix:\n{conf_mat}")
    print(f"Accuracy = {accuracy}")

    pickle.dump(tfidf_vectorizer, open(TFIDF_FILENAME, 'wb'))
    pickle.dump(model, open(model_filename, 'wb'))


def evaluate(test_features, model, algorithm):
    if algorithm == 'knn':
        y_hat = model.predict(test_features)
    elif algorithm == 'dnn':
        model.eval()
        y_hat = model(torch.from_numpy(test_features).float())
        y_hat = y_hat.detach().numpy() >= 0.5
    return y_hat


def predict(input_sentence, model, algorithm):
    if algorithm == 'knn':
        return model.predict(input_sentence)
    elif algorithm == 'dnn':
        return model(torch.from_numpy(input_sentence).float())


def infer(algorithm):
    tfidf_vectorizer = preprocessing.load_vectorizer()
    if algorithm == 'knn':
        model_filename = KNN_FILENAME
    elif algorithm == 'dnn':
        model_filename = DNN_FILENAME
    model = pickle.load(open(model_filename, 'rb'))

    # sample_phrase = "Horrible batting by England, bad match"
    sample_phrase = "Excellent performance by India in the match, beautiful game"
    sample_phrase_vectorized = preprocess_sentence_for_inference(sample_phrase,
                                                                 stemmer=PorterStemmer(),
                                                                 vectorizer=tfidf_vectorizer)
    prediction = predict(sample_phrase_vectorized, model, algorithm)
    print(f"Sample sentence: {sample_phrase}")
    print(f"Vectorized sentence: {sample_phrase_vectorized}")
    print(f"Prediction: {prediction}")


if __name__ == '__main__':
    should_train = True
    algorithm = 'dnn'
    if should_train:
        train(algorithm)
    else:
        infer(algorithm)
