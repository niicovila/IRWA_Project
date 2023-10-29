
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
    tf = defaultdict(list) 
    df = defaultdict(int)  
    idf = defaultdict(float)

    index = defaultdict(list)
    num_documents = len(lines)
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

        norm = 0
        for term, posting in current_page_index.items():
            # posting will contain the list of positions for current term in current document.
            # posting ==> [current_doc, [list of positions]]
            # you can use it to infer the frequency of current term.
            norm += len(posting[1]) ** 2
        norm = math.sqrt(norm)

        #calculate the tf(dividing the term frequency by the above computed norm) and df weights
        for term, posting in current_page_index.items():
            # append the tf for current term (tf = term frequency in current doc/norm)
            tf[term].append([posting[0], np.round(len(posting[1]) / norm, 4)]) ## SEE formula (1) above
            #increment the document frequency of current term (number of documents containing the current term)
            df[term] += 1 # increment DF for current term

        #merge the current page index with the main index
        for term_page, posting_page in current_page_index.items():
            index[term_page].append(posting_page)

        # Compute IDF following the formula (3) above. HINT: use np.log
        for term in df:
            idf[term] = np.round(np.log(float(num_documents / df[term])), 4)
    return index, tf, df, idf

def select_docs(data, query_id):
    subset = []
    ground_truths = []

    for line in data:
        doc, q_id, label = line.split(',')
        if q_id == query_id:
            subset.append(doc)
            if label == '1':
                ground_truths.append(1)
            else:
                ground_truths.append(0)
        elif label == '1':
            subset.append(doc)
            ground_truths.append(0)

    return subset, ground_truths

def read_csv(path):
    with open(path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Salta el encabezado
        data = [",".join(row) for row in reader]
    return data

def precision_at_k(doc_score, y_score, k=10):
    """
    Parameters
    ----------
    doc_score: Ground truth (true relevance labels).
    y_score: Predicted scores.
    k : number of doc to consider.

    Returns
    -------
    precision @k : float

    """
    order = np.argsort(y_score)[::-1]
    doc_score = np.take(doc_score, order[:k])
    relevant = sum(doc_score == 1)
    return float(relevant) / k


def evaluation(queries, i, lines, tweet_text, tweet_document_ids_map, path):
    evaluation_data1 = read_csv(path)
    docs, ground_truths = select_docs(evaluation_data1,f"Q{i}")

    subset = [line for line in lines if tweet_document_ids_map[json.loads(line)["id"]] in(docs)]
    subset = sorted(subset, key=lambda line: docs.index(tweet_document_ids_map[json.loads(line)["id"]]))
 
    subset_tweets_ids = [json.loads(line)["id"] for line in subset]
    subindex, subtf, subdf, subidf = create_index(subset)

    results, scores = search_tf_idf(queries[i-1], subindex, subidf, subtf)
    
    y_scores = [scores[results.index(tweet)] if(tweet in results) else 0 for tweet in subset_tweets_ids]
    relevant_tweets = tweet_text[tweet_text["tweet_id"].isin(results)]

    tweet_dict = {row["tweet_id"]: row["text"] for _, row in tweet_text.iterrows()}
    relevant_tweets = [tweet_dict[tweet_id] for tweet_id in results]
    file_path = f'relevant_text_q{i}.txt'
    with open(file_path, 'w', encoding="utf-8") as file:
        file.write("\n\n\n".join(relevant_tweets))

    precision= precision_at_k(ground_truths, y_scores)
    print(f'Query: {queries[i-1]}; Precision: {precision}')
    return precision

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

def rank_documents(terms, docs, index, idf, tf):
    """
    Perform the ranking of the results of a search based on the tf-idf weights

    Argument:
    terms -- list of query terms
    docs -- list of documents, to rank, matching the query
    index -- inverted index data structure
    idf -- inverted document frequencies
    tf -- term frequencies
    title_index -- mapping between page id and page title

    Returns:
    Print the list of ranked documents
    """
    doc_vectors = defaultdict(lambda: [0] * len(terms)) # I call doc_vectors[k] for a nonexistent key k, the key-value pair (k,[0]*len(terms)) will be automatically added to the dictionary
    query_vector = [0] * len(terms)


    query_terms_count = collections.Counter(terms)  # get the frequency of each term in the query.

    query_norm = la.norm(list(query_terms_count.values()))

    for termIndex, term in enumerate(terms):  #termIndex is the index of the term in the query
        if term not in index:
            continue
        query_vector[termIndex] = query_terms_count[term] / query_norm * idf[term]

        # Generate doc_vectors for matching docs
        for doc_index, (doc, postings) in enumerate(index[term]):
            # Example of [doc_index, (doc, postings)]
            # 0 (26, array('I', [1, 4, 12, 15, 22, 28, 32, 43, 51, 68, 333, 337]))
            # 1 (33, array('I', [26, 33, 57, 71, 87, 104, 109]))
            # term is in doc 26 in positions 1,4, .....
            # term is in doc 33 in positions 26,33, .....

            #tf[term][0] will contain the tf of the term "term" in the doc 26
            if doc in docs:
                doc_vectors[doc][termIndex] = tf[term][doc_index][1] * idf[term]

    doc_scores = [[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items()]
    doc_scores.sort(reverse=True)

    result_docs = [x[1] for x in doc_scores]
    result_scores = [x[0] for x in doc_scores]
    
    if len(result_docs) == 0:
        print("No results found, try again")
        query = input()
        docs = search_tf_idf(query, index)

    return result_docs, result_scores


def search_tf_idf(query, index, idf, tf):
    """
    output is the list of documents that contain any of the query terms.
    So, we will get the list of documents for each query term, and take the union of them.
    """
    query = build_terms(query)
    docs = set()
    for term in query:
        try:
            # store in term_docs the ids of the docs that contain "term"
            term_docs = [posting[0] for posting in index[term]]

            # docs = docs Union term_docs
            docs |= set(term_docs)
        except:
            #term is not in index
            pass
    docs = list(docs)
    ranked_docs, scores = rank_documents(query, docs, index, idf, tf)
    return ranked_docs, scores


def main():
    docs_path = '/Users/nvila/Downloads/Rus_Ukr_war_data.json'
    with open(docs_path) as fp:
        lines = fp.readlines()
    lines = [l.strip().replace(' +', ' ') for l in lines]

    ids_path = '/Users/nvila/Downloads/ids.csv'
    doc_ids = pd.read_csv(ids_path, sep='\t', header=None)
    
    doc_ids.columns = ["doc_id", "tweet_id"]
    tweet_document_ids_map = {}
    for index, row in doc_ids.iterrows():
        tweet_document_ids_map[row['tweet_id']] = row['doc_id']

    tweet_text = pd.DataFrame({'tweet_id': [json.loads(line)["id"] for line in lines], 'text': [json.loads(line)["full_text"] for line in lines]})


    baseline_queries = [
        "Tank Kharkiv",
        "Nord Stream pipeline",
        "Annexation territories Russia"
    ]

    custom_queries = [
        "Russian military intervention in Ukraine",
        "Impact of sanctions on Russia",
        "Russian propaganda in the Ukraine conflict",
        "International response to Russia-Ukraine war",
        "Humanitarian crisis in Ukraine"
    ]
    ev1 = '/Users/nvila/Downloads/Evaluation_gt.csv'
    ev2 = '/Users/nvila/Downloads/evaluation_custom_queries.csv'

    b = [evaluation(baseline_queries, i+1, lines, tweet_text, tweet_document_ids_map, path = ev1) for i in range(len(baseline_queries))]
    c = [evaluation(custom_queries, i+1, lines, tweet_text, tweet_document_ids_map, path = ev2) for i in range(len(custom_queries))]

main()