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
import csv
import random
import pandas as pd
import matplotlib.pyplot as plt
from torch import cosine_similarity
from wordcloud import WordCloud
import nltk
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
nltk.download('stopwords')
from utils import build_terms, read_tweets
from gensim.models import Word2Vec
from index import create_index, search_tf_idf, rank_documents

def query():
    docs_path = '/Users/nvila/Downloads/Rus_Ukr_war_data.json'
    with open(docs_path) as fp:
        lines = fp.readlines()
    lines = [l.strip().replace(' +', ' ') for l in lines]
    print("There are ", len(lines), " tweets about the Russia-Ukraine War")
    

    # Process lines to create a list of tweet IDs
    tweet_ids = [json.loads(line)["id"] for line in lines]
    tweet_ids_df = pd.DataFrame({'tweet_id': tweet_ids, 'position': list(range(len(tweet_ids)))})
    tweet_text = pd.DataFrame({'tweet_id': [json.loads(line)["id"] for line in lines], 'text': [json.loads(line)["full_text"] for line in lines]})
    index, tf, df, idf = create_index(lines)
    query = '0'
    while query != '1':
        print("Write your query here: (Press 1 to exit))\n")
        query = input()
        if query != '1':
            results, scores = search_tf_idf(query, index, idf, tf)
            relevant_tweets = tweet_text[tweet_text["tweet_id"].isin(results)]

            tweet_dict = {row["tweet_id"]: row["text"] for _, row in tweet_text.iterrows()}
            relevant_tweets = [tweet_dict[tweet_id] for tweet_id in results]
            file_path = f'relevant_text_user_query.txt'
            with open(file_path, 'w', encoding="utf-8") as file:
                file.write("\n\n\n".join(relevant_tweets))
            print("Your results have been saved")

if __name__ == '__main__':
    query()

