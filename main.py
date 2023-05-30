import streamlit as st
from tweets import scrape_tweets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
import pandas as pd
import torch

st.title('Enter Your Twitter Account-')
query = st.text_input("Enter account: ")
limit = 20

def predict_sentiment(query):
    if query:
        df = pd.DataFrame(scrape_tweets(query,limit), columns=['Date', 'User', 'Tweet'])
        # st.write(df)

        roberta = "cardiffnlp/twitter-roberta-base-sentiment"
        model = AutoModelForSequenceClassification.from_pretrained(roberta)
        tokenizer = AutoTokenizer.from_pretrained(roberta)
        labels = ['Negative', 'Neutral', 'Positive']

        def sentiment_score(x):
            tweet_words = []
            for word in x.split(' '):
                if word.startswith('@') and len(word) > 1:
                    word = '@user'
                elif word.startswith('http'):
                    word = "http"
                tweet_words.append(word)

            tweet_proc = " ".join(tweet_words)

            encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
            output = model(**encoded_tweet)

            scores = output.logits[0].detach().numpy()
            scores = softmax(scores)

            max_score = np.argmax(scores)
            sentiment = labels[max_score]

            return sentiment

        df['sentiment'] = df['Tweet'].apply(sentiment_score)
        sentiment_counts = df['sentiment'].value_counts()
        negative_count = sentiment_counts.get('Negative', 0)
        neutral_count = sentiment_counts.get('Neutral', 0)
        positive_count = sentiment_counts.get('Positive', 0)

        st.write("Neutral:", (neutral_count/limit)*100,"%")
        st.write("Positive:", (positive_count/limit)*100,"%")
        st.write("Negative:", (negative_count/limit)*100,"%")


trigger = st.button('Predict', on_click=lambda: predict_sentiment(query))
