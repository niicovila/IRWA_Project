import nltk
nltk.download('stopwords')
from utils import build_terms, read_tweets, get_tweet
from index import create_index, search_tf_idf, rank_documents

def query():
    docs_path = 'IRWA_Project\Rus_Ukr_war_data.json'
    tweets = read_tweets(docs_path)
    index, tf, df, idf = create_index(tweets)
    query = '0'
    while query != '1':
        print("Write your query here: (Press 1 to exit))\n")
        query = input()
        if query != '1':
            results, scores = search_tf_idf(query, index, idf, tf)
            print(len(results))
            file_path = f'relevant_text_user_query.txt'
            
            with open(file_path, 'w', encoding='utf-8') as output_file:
                for tweet_id in results[:20]:
                    _, selected_tweet = get_tweet(tweet_id, tweets)
                    if selected_tweet:
                        output_file.write(selected_tweet + '\n')
                print(f"Tweet information has been written to {file_path}")

if __name__ == '__main__':
    query()

