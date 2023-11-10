from utils import get_tweet, read_tweets
from index import build_terms, create_index
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
    # Use word2vec model with the tweets, calculating the average vector of all the terms in each tweet
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

def create_vector_structure(tweets, ids, model):
    vectors = []
    for tweet in tweets:
        vectors.append(get_tweet_vector(tweet, model))
    return pd.DataFrame({'id': [id for id in ids], 'vector': [v for v in vectors]})

def rank_tweets_cosine(query, tweets_df, model, k):
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

def get_matching_tweets(query, tweets, index):
    query_terms = build_terms(query)
    matching_tweet_ids = None
    
    # Find tweets that contain all the query terms
    for term in query_terms:
        if term in index:
            tweet_ids_with_term = set([doc[0] for doc in index[term]])
            if matching_tweet_ids is None:
                matching_tweet_ids = tweet_ids_with_term
            else:
                matching_tweet_ids = matching_tweet_ids.intersection(tweet_ids_with_term)
        else:
            # If any term is not found in the index, there won't be matching tweets
            matching_tweet_ids = set()
            break

    matching_tweet_ids = list(matching_tweet_ids)
    tweet_texts = []
    # Retrieve tweet texts corresponding to the matching tweet IDs
    for tweet_id in matching_tweet_ids:
        tweet, _ = get_tweet(tweet_id, tweets)
        tweet_texts.append(tweet['Tweet'])
    
    return matching_tweet_ids, tweet_texts

if __name__ == "__main__":

    docs_path = '/Users/nicolasvila/workplace/uni/IRWA_Project/Rus_Ukr_war_data.json'
    tweets = read_tweets(docs_path)
    model = get_word2vec_model(tweets)  # The tweet2vec model
    index, _, _, _ = create_index(tweets)
    all_tweets_text = [json.loads(tweet)['full_text'] for tweet in tweets]
    all_tweet_ids = [json.loads(tweet)['id'] for tweet in tweets]
    tweets_df = create_vector_structure(all_tweets_text, all_tweet_ids, model)

    while True:
        query = input('Enter your query (type "exit" to end): ')
        if query.lower() == 'exit':
            break
        ids, tweets_text = get_matching_tweets(query, tweets, index)
        if not ids:
            print("No matching tweets found for the query.")
        else:
            subset_df = tweets_df.loc[tweets_df['id'].isin(ids)]
            retrieved_tweets = rank_tweets_cosine(query, subset_df, model, k=20)
            output_file_path = './relevant_tweets.txt'
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                for tweet_id in retrieved_tweets:
                    _, selected_tweet = get_tweet(tweet_id, tweets)
                    if selected_tweet:
                        output_file.write(selected_tweet + '\n')
                print(f"Tweet information has been written to {output_file_path}")

    