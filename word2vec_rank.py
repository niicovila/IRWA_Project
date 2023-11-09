from utils import get_tweet, read_tweets
from index import build_terms
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


def get_word2vec_model(lines):
    tweets = []
    for line in lines:
        tweet = json.loads(line)
        terms_in_tweet = build_terms(tweet["full_text"])
        tweets.append(terms_in_tweet)
    #Use word2vec model with the tweets, calculating the average vector of all the terms in each tweet
    model = Word2Vec(sentences=tweets, workers=4, min_count=1, window=10, sample=1e-3)
    return model

def get_tweet_vector(tweet, model):
    tweet = build_terms(tweet)
    vectorized_terms = [model.wv[word] for word in tweet if word in model.wv]
    if vectorized_terms:
        tweet_vector = np.mean(vectorized_terms, axis=0)
        return tweet_vector
    else:
        return None

def create_vector_structure(tweets, model):
    vectors = []
    ids = []
    for tweet in tweets:
        tweet = json.loads(tweet)
        tweet_text = tweet['full_text']
        ids.append(tweet['id'])
        vectors.append(get_tweet_vector(tweet_text, model))
    return pd.DataFrame({'id':[id for id in ids], 'vector': [v for v in vectors]})

def rank_tweets(query, tweets_df, model, k):
    query_vector = get_tweet_vector(query, model)
    if query_vector is not None:
        tweet_vectors = np.array(tweets_df['vector'].tolist())
        cosine_similarities = cosine_similarity([query_vector], tweet_vectors)[0]
        top_indices = np.argsort(cosine_similarities)[-k:][::-1]    # Get indices of top k tweets based on cosine similarity
        top_tweet_ids = tweets_df.iloc[top_indices]['id'].tolist()
        return top_tweet_ids
    else:
        print("Error: Unable to generate vector for the query.")
        return []


if __name__ == "__main__":
    docs_path = '/Users/nicolasvila/workplace/uni/IRWA_Project/Rus_Ukr_war_data.json'
    tweets = read_tweets(docs_path)
    tweet_id = 1575918221013979136  # Replace this with the desired tweet ID
    selected_tweet = get_tweet(tweet_id, tweets)

    if selected_tweet:
        print(selected_tweet)
    else:
        print("Tweet not found.")
    

    model = get_word2vec_model(tweets)
    # tweets_df = create_vector_structure(tweets, model)

    # query = 'russia putin war'
    # tweets = rank_tweets(query, tweets_df, model, k=10)