import argparse

import torch
from nltk import PorterStemmer
from classifiers import dnn
import preprocessing
from twitter import TwitterClient


def run(keyword: str):
    twitter_client = TwitterClient("9mCSzfgGPADID9d7BEu5LKNqz",
                                   "YNmuDCETooxcvyZLxkJGEH1y81yIPHQP3QECqft6LIYcN8Qp1D",
                                   "AAAAAAAAAAAAAAAAAAAAANpOfgEAAAAA1xyBAkg9x08vTUpGCLkzUjd90Wk%3DwrJ0zEZCUEQvtoqbrf4TcRMqqQMAvD9gx9TzwdCfiVPT7HMwve")
    sentiment_classifier = dnn.load_model()
    sentiment_classifier.eval()
    stemmer = PorterStemmer()
    vectorizer = preprocessing.load_vectorizer()

    tweets = twitter_client.fetch_tweets_containing(keyword, 100)
    tweets_processed = list(map(lambda tweet: preprocessing.preprocess_sentence_for_inference(tweet, stemmer, vectorizer), tweets))

    results = []
    for tweet_raw, tweet_processed in zip(tweets, tweets_processed):
        results.append((tweet_raw,
                        sentiment_classifier(torch.from_numpy(tweet_processed).float()).detach().numpy()[0][0]))

    negative_tweets = list(filter(lambda result: result[1] < 0.5, results))
    positive_tweets = [element for element in results if element[1] >= 0.5]

    print(f"Found {len(negative_tweets)} negative tweets and {len(positive_tweets)} positive ones out of {len(tweets)}.")
    print(f"Positives:\n{positive_tweets}")
    print(f"Negatives:\n{negative_tweets}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--keyword", required=True, type=str)
    args = parser.parse_args()

    keyword = args.keyword
    run(keyword)