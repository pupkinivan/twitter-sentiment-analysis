# Twitter sentiment analysis

Based on a lecture from [an Udemy course](https://www.udemy.com/course/machine-learning-deep-learning-model-deployment/), I refactored the logic to offer KNN- and DNN-based binary classification of tweets: whether their content is negative or positive.

The contents of each tweet are vectorized with `sklearn`'s TF-IDF, after the stem of each word was extracted with `nltk`'s `PorterStemmer`.

The training data was provided in the course, and is comprised by restaurant reviews (as can be seen in `./resources/Restaurant_Reviews.tsv`).

The models can be trained in the `train_models.py` file. Inference can be done with it, too, by hardcoding a sample sentence.

Tweets can be pulled and analyzed with the `pull_analyzed.py` file, querying based on a keyword given in a required command line argument `--keyword`.