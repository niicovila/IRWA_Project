import nltk
nltk.download('stopwords')
from utils import build_terms, read_tweets, get_tweet
from index import create_index, search_tf_idf, rank_documents
import pandas as pd
from embeddings_score import obtain_similarity, create_embeddings, parse

def query():
    k = 20
    docs_path = 'IRWA_Project\Rus_Ukr_war_data.json'
    print('Reading embeddings...')
    combined_df = pd.read_csv('IRWA_Project\embeddings_df.csv')
    combined_df["Embedding"] = combined_df["Embedding"].apply(lambda vector_string: parse(vector_string))
    print('Embeddings loaded!')
    tweets = read_tweets(docs_path)
    print('Creating index...')
    index, tf, df, idf = create_index(tweets)
    print('Index crated!')
    query = '0'
    while query != '1':
        print("\nWrite your query here: (Press 1 to exit))\n")
        query = input()
        if query != '1':
            results1, scores1 = search_tf_idf(query, index, idf, tf)
            results2 = obtain_similarity(query, combined_df, k)
            print('Saving the results...')
            file_path1 = f'ranking_tfidf.txt'
            file_path2 = f'ranking_embeddings.txt'
            
            with open(file_path1, 'w', encoding='utf-8') as output_file:
                for tweet_id in results1[:k]:
                    _, selected_tweet = get_tweet(tweet_id, tweets)
                    if selected_tweet:
                        output_file.write(selected_tweet + '\n')
                print(f"Tweet tf-idf ranking information has been written to {file_path1}")

            with open(file_path2, 'w', encoding='utf-8') as output_file:
                for tweet_id in results2['Tweet_id'][:k]:
                    _, selected_tweet = get_tweet(tweet_id, tweets)
                    if selected_tweet:
                        output_file.write(selected_tweet + '\n')
                print(f"Tweet custom information has been written to {file_path2}")

if __name__ == '__main__':
    query()