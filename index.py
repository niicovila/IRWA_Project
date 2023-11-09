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
import csv
import random
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import build_terms, read_tweets
from gensim.models import Word2Vec
nltk.download('stopwords')

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

def recall_at_k(doc_score, y_score, k=10):
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
    r = np.sum(doc_score)
    order = np.argsort(y_score)[::-1]
    doc_score = np.take(doc_score, order[:k])
    relevant = sum(doc_score == 1)
    return float(relevant) / r

def avg_precision_at_k(doc_score, y_score, k=10):
    """
    Parameters
    ----------
    doc_score: Ground truth (true relevance labels).
    y_score: Predicted scores.
    k : number of doc to consider.

    Returns
    -------
    average precision @k : float
    """
    gtp = np.sum(doc_score)
    order = np.argsort(y_score)[::-1]
    doc_score = np.take(doc_score, order[:k])
    ## if all documents are not relevant
    if gtp == 0:
        return 0
    n_relevant_at_i = 0
    prec_at_i = 0
    for i in range(len(doc_score)):
        if doc_score[i] == 1:
            n_relevant_at_i += 1
            prec_at_i += n_relevant_at_i / (i + 1)
    return prec_at_i / gtp

def dcg_at_k(doc_score, y_score, k=10):
    order = np.argsort(y_score)[::-1]  # get the list of indexes of the predicted score sorted in descending order.
    doc_score = np.take(doc_score, order[:k])  # sort the actual relevance label of the documents based on predicted score(hint: np.take) and take first k.
    gain = 2 ** doc_score - 1  # Compute gain (use formula 7 above)
    discounts = np.log2(np.arange(len(doc_score)) + 2)  # Compute denominator
    return np.sum(gain / discounts)  #return dcg@k


def ndcg_at_k(doc_score, y_score, k=10):
    dcg_max = dcg_at_k(doc_score, doc_score, k)
    if not dcg_max:
        return 0
    return np.round(dcg_at_k(doc_score, y_score, k) / dcg_max, 4)

def rr_at_k(doc_score, y_score, k=10):
    """
    Parameters
    ----------
    doc_score: Ground truth (true relevance labels).
    y_score: Predicted scores.
    k : number of doc to consider.

    Returns
    -------
    Reciprocal Rank for qurrent query
    """

    order = np.argsort(y_score)[::-1]  # get the list of indexes of the predicted score sorted in descending order.
    doc_score = np.take(doc_score, order[
                             :k])  # sort the actual relevance label of the documents based on predicted score(hint: np.take) and take first k.
    if np.sum(doc_score) == 0:  # if there are not relevant doument return 0
        return 0
    return 1 / (np.argmax(doc_score == 1) + 1)  # hint: to get the position of the first relevant document use "np.argmax"



def evaluation(queries, i, lines, tweet_text, tweet_document_ids_map, k, custom, path):
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

    if(custom):
        file_path = f'text_custom_q{i}.txt'
    else: 
        file_path = f'text_baseline_q{i}.txt'
    with open(file_path, 'w', encoding="utf-8") as file:
        file.write("\n\n\n".join(relevant_tweets))

    print(f'\nQuery {i}: {queries[i-1]}\n')

    precision= precision_at_k(ground_truths, y_scores)
    print(f'Precision at {k}: {precision}')

    recall = recall_at_k(ground_truths, y_scores)
    print(f'Recall at {k}: {recall}')

    avg_precision = avg_precision_at_k(ground_truths, y_scores)
    print(f'Average precision at {k}: {avg_precision}')

    fscore = (2*recall*precision)/(recall+precision)
    print(f'F1-Score at {k}:  {fscore}')

    ndcg = ndcg_at_k(ground_truths, y_scores)
    print(f'NDG at {k}:  {ndcg}')

    rr = rr_at_k(ground_truths, y_scores)

    return avg_precision, rr

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
        result_docs, result_scores = search_tf_idf(query, index, idf, tf)

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
def plot_tnse(lines):
    tweets = []
    for line in lines:
        tweet = json.loads(line)
        terms_in_tweet = build_terms(tweet["full_text"])
        tweets.append(terms_in_tweet)
    #Use word2vec model with the tweets, calculating the average vector of all the terms in each tweet
    model = Word2Vec(sentences=tweets, workers=4, min_count=1, window=10, sample=1e-3)
    tweet_vectors = []
    for terms in tweets:
        vectorized_terms = [model.wv[word] for word in terms if word in model.wv]
        if vectorized_terms:
            tweet_vector = np.mean(vectorized_terms, axis=0)  # Average the word vectors to get a single vector per tweet
            tweet_vectors.append(tweet_vector)

    X = np.array(tweet_vectors)
    print(len(X))
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)

    # Plot the t-SNE representation
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    plt.show()

def main():
    docs_path = '/Users/nvila/Downloads/Rus_Ukr_war_data.json'
    # docs_path = '/Users/guill/OneDrive/Escritorio/Rus_Ukr_war_data.json'

    ids_path = '/Users/nvila/Downloads/ids.csv'
    # ids_path = '/Users/guill/OneDrive/Escritorio/Rus_Ukr_war_data_ids.csv'

    ev1 = '/Users/nvila/Downloads/Evaluation_gt.csv'
    ev2 = '/Users/nvila/Downloads/evaluation_custom_queries.csv'

    # ev1 = '/Users/guill/OneDrive/Escritorio/Evaluation_gt.csv'
    # ev2 = '/Users/guill/OneDrive/Escritorio/evaluation_custom_queries.csv'

    with open(docs_path) as fp:
        lines = fp.readlines()
    lines = [l.strip().replace(' +', ' ') for l in lines]

    

    doc_ids = pd.read_csv(ids_path, sep='\t', header=None)
    
    doc_ids.columns = ["doc_id", "tweet_id"]
    tweet_document_ids_map = {}
    for index, row in doc_ids.iterrows():
        tweet_document_ids_map[row['tweet_id']] = row['doc_id']

    tweet_text = pd.DataFrame({'tweet_id': [json.loads(line)["id"] for line in lines], 'text': [json.loads(line)["full_text"] for line in lines]})


    baseline_queries = [
        "Tank Kharkiv",
        "Nord Stream pipeline",
        "Annexation territories"
    ]

    custom_queries = [
        "Russian military intervention",
        "Impact of sanctions on Russia",
        "Russian propaganda in the conflict",
        "International response to Russia-Ukraine war",
        "Humanitarian crisis"
    ]

    k = 10

    n_baseline = len(baseline_queries)
    MAP_baseline = 0
    MRR_baseline = 0
    for i in range(len(baseline_queries)):
        AP_baseline, RR_baseline = evaluation(baseline_queries, i+1, lines, tweet_text, tweet_document_ids_map, k, custom=False, path = ev1)
        MAP_baseline += AP_baseline
        MRR_baseline += RR_baseline
    print(f'\nMean Average Precision of Baseline Queries: {MAP_baseline/n_baseline}')
    print(f'Mean Reciprocal Rank of Baseline Queries: {MRR_baseline/n_baseline}')
    
    n_custom = len(custom_queries)
    MAP_custom = 0
    MRR_custom = 0
    for i in range(len(custom_queries)):
        AP_custom , RR_custom  = evaluation(custom_queries, i+1, lines, tweet_text, tweet_document_ids_map, k, custom=True, path = ev2)
        MAP_custom  += AP_custom 
        MRR_custom  += RR_custom 
    print(f'\nMean Average Precision of Custom Queries: {MAP_custom /n_custom }')
    print(f'Mean Reciprocal Rank of Custom Queries: {MRR_custom /n_custom }')

    plot_tnse(lines)

if __name__ == '__main__':
    main()