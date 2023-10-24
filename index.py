import time
import json
from collections import defaultdict
from array import array
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import math
import numpy as np
import collections
from numpy import linalg as la
import string
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
nltk.download('stopwords')
from utils import build_terms, read_tweets

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


if "name" == "__main__":
    file_path = ''
    start_time = time.time()
    lines = []
    index = create_index(lines)
    print("Total time to create the index: {} seconds".format(np.round(time.time() - start_time, 2)))

    print("Index results for the term 'putin': {}\n".format(index['putin']))
    print("First 10 Index results for the term 'putin': \n{}".format(index['putin'][:10]))