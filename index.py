import time
import json
from collections import Counter, defaultdict
from array import array
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import math
import numpy as np
import collections
from numpy import linalg as la
import string
from openai.embeddings_utils import cosine_similarity
import re
import pandas as pd
import matplotlib.pyplot as plt
from torch import cosine_similarity
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer
import nltk
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
nltk.download('stopwords')
from utils import build_terms, read_tweets
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def create_index(lines):
    """
    Implement the inverted index

    Argument:
    lines -- collection of Wikipedia articles

    Returns:
    index - the inverted index (implemented through a Python dictionary) containing terms as keys and the corresponding
    list of documents where these keys appears in (and the positions) as values.
    """
    index = defaultdict(list)
    for line in lines:  # Remember, lines contain all documents: article-id | article-title | article-body
       
        line = json.loads(line)
        line_arr = line["full_text"]
        tweet_id = line["id"]  # Get the tweet ID
        terms = build_terms(line_arr)

        current_page_index = {}

        for position, term in enumerate(terms): # terms contains page_title + page_text. Loop over all terms
            try:
                current_page_index[term][tweet_id].append(position)

            except:
               
                current_page_index[term] = [tweet_id, array('I', [position])]

        #merge the current page index with the main index
        for term_page, posting_page in current_page_index.items():
            index[term_page].append(posting_page)

        ## END CODE

    return index

def vector_index(lines):
    
    """
    input: list of paragraphs
    output: dataframe mapping each paragraph to its embedding
    """
    # from sklearn.cluster import AgglomerativeClustering
    embeddings = model.encode(lines)
    df = pd.DataFrame(
        {"tweet": lines[i]["id"], "vector_representation": embeddings[i]}
        for i in range(len(embeddings))
    )
    return df
def obtain_similarity(query, df, k):
    """
    arguments:
        - query: word or sentence to compare
        - df: dataframe mapping paragraphs to embeddings
        - k: number of selected similar paragraphs
    output: list of paragraphs relevant for the query and the position in the datframe at which they are
    """

    query_embedding = model.encode(query)
    df["similarity"] = df["vector_representation"].apply(
        lambda x: cosine_similarity(x, query_embedding)
    )
    results = df.sort_values("similarity", ascending=False, ignore_index=True)
    top_k = results["tweet"][1:k]
    top_k = list(top_k)
    ## Find positions of the top_k in df
    positions = df.loc[df["tweet"].isin(top_k)].index
    return top_k, positions

def calculate_tf_idf(index):
    tf_idf_scores = {}
    total_tweets = len(index.keys())

    # Calculate IDF for each term
    idf = {term: np.log(total_tweets / len(postings)) for term, postings in index.items()}

    # Calculate TF-IDF scores for each term in each tweet
    for term, postings in index.items():
        tf_idf_scores[term] = {}
        for posting in postings:
            tweet_id, positions = posting[0], posting[1]
            tf = len(positions)
            tf_idf = tf * idf[term]
            tf_idf_scores[term][tweet_id] = tf_idf

    return tf_idf_scores
def scatter_plot(df):
    # Apply T-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    tweet_tsne = tsne.fit_transform(df.vector_representation.values())

    # Plot the tweets in a scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(tweet_tsne[:, 0], tweet_tsne[:, 1])
    plt.title('T-SNE Visualization of Tweets')
    plt.xlabel('T-SNE Component 1')
    plt.ylabel('T-SNE Component 2')
    plt.savefig("./scatter_plot")
    plt.close()  # Close the plot to release resources

def retrieve_top_k_tweets(query, index, tf_idf_scores, k=10):
    query_terms = build_terms(query)  # Tokenize the query
    query_tf_idf = Counter(query_terms)

    # Calculate TF-IDF scores for the query
    query_scores = {term: query_tf_idf[term] * np.log(len(index) / len(index[term])) for term in query_terms}

    # Compute cosine similarity between query and tweets
    cosine_similarities = {}
    for term, score in query_scores.items():
        for tweet_id, tf_idf in tf_idf_scores.get(term, {}).items():
            if tweet_id not in cosine_similarities:
                cosine_similarities[tweet_id] = 0
            cosine_similarities[tweet_id] += score * tf_idf

    # Rank tweets based on cosine similarity
    ranked_tweets = [tweet_id for tweet_id, similarity in sorted(cosine_similarities.items(),
                                                                 key=lambda x: x[1], reverse=True)[:k]]

    return ranked_tweets

if "name" == "__main__":
    file_path = ''
    start_time = time.time()
    lines = []
    index = create_index(lines)
    print("Total time to create the index: {} seconds".format(np.round(time.time() - start_time, 2)))

    print("Index results for the term 'putin': {}\n".format(index['putin']))
    print("First 10 Index results for the term 'putin': \n{}".format(index['putin'][:10]))

    query = "putin Russia"

    # Calculate TF-IDF scores
    tf_idf_scores = calculate_tf_idf(index)

    # Retrieve top k relevant tweets for the query
    k = 10  # Number of top tweets to retrieve
    relevant_tweets = retrieve_top_k_tweets(query, index, tf_idf_scores, k)

    # Print the relevant tweets
    print("Top {} Relevant Tweets for the Query '{}':".format(k, query))
    for tweet_id in relevant_tweets:
        print("Tweet ID:", tweet_id) 
    df = vector_index(lines)
    scatter_plot(df)