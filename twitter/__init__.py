import datetime

import tweepy


class TwitterClient:
    def __init__(self, api_key: str, secret_key: str, bearer_token: str):
        self._api_key = api_key
        self._secret_key = secret_key
        self._bearer_token = bearer_token
        self._oauth = tweepy.OAuthHandler(api_key, secret_key, access_token=bearer_token, access_token_secret=secret_key)
        self._oauth2_bearer = tweepy.OAuth2BearerHandler(bearer_token)
        self._oauth2_app = tweepy.OAuth2AppHandler(api_key, secret_key)
        self._http = tweepy.Client(bearer_token=bearer_token, consumer_key=api_key, consumer_secret=secret_key)

    def fetch_tweets_containing(self, keyword: str, limit: int = 100, start_time: str = None):
        if start_time is None:
            start_time = datetime.datetime.now() - datetime.timedelta(days=6, hours=12)
        start_time = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"Start time for pulling tweets: {start_time}")

        tweets_list = []
        response_data = self._http.search_recent_tweets(keyword, max_results=limit, start_time=start_time).data
        for item in response_data:
            tweets_list.append(item.text)
        return tweets_list