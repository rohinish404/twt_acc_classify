import snscrape.modules.twitter as sntwitter
import pandas as pd

def scrape_tweets(query, limit=10):
    tweets = []
    for tweet in sntwitter.TwitterProfileScraper(query).get_items():
        if len(tweets) == limit:
            break
        else:
            tweets.append([tweet.date, tweet.username, tweet.content])
    return tweets