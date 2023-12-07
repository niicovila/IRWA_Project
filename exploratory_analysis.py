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
import spacy
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
nltk.download('stopwords')
from utils import build_terms, read_tweets
nlp = spacy.load("en_core_web_sm")

def build_and_filter_histogram(words_dict, count_threshold):
        filtered_words = {word: count for word, count in words_dict.items() if count >= count_threshold}
        sorted_filtered_words = dict(sorted(filtered_words.items(), key=lambda item: item[1], reverse=True))

        plt.figure(figsize=(10, 6))
        plt.bar(sorted_filtered_words.keys(), sorted_filtered_words.values(), color='skyblue')
        plt.xlabel('Words')
        plt.ylabel('Count')
        plt.title(f'Word Frequency Histogram (Count Threshold: {count_threshold})')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig("./histogram_word_count")
        plt.close()  # Close the plot to release resources

def perform_entity_recognition(text):
        doc = nlp(text)
        if doc.ents:
            entities = [ent.label_ for ent in doc.ents]
            return entities
        return 0

def exploratory_analysis(lines):
    words = {}
    processed_lines = []
    tweet_document_ids_map = {}

    for line in lines:
        line = json.loads(line)
        line_arr = line["full_text"]
        tweet_id = line["id"]  # Get the tweet ID
        line_arr = build_terms(line_arr)
        processed_lines.append(line_arr)
        # Map the tweet ID to the document ID (tweet ID can be used as the document ID in this case)
        tweet_document_ids_map[tweet_id] = tweet_id
        #print(line_arr)
        for term in line_arr:
            if term not in words:
                words[term] = 1
            else:
                words[term] += 1

    print("There are {} different words in the dataset.".format(len(words.keys()))+ '\n')
    build_and_filter_histogram(words, 1000)


    wc = WordCloud(width=800, height=800, background_color='black', colormap='viridis')

    # Generate the word cloud from the word frequencies
    wc.generate_from_frequencies(words)
    wc.to_file( './word_cloud.png')



    tweet_lengths_words = [len(line) for line in processed_lines]
    tweet_lengths_chars = [len(json.loads(line)['full_text']) for line in lines]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns
    axs[0].hist(tweet_lengths_words, bins=50, align='left', color='skyblue', edgecolor='black')
    axs[0].set_xlabel('Number of Words')
    axs[0].set_ylabel('Number of Tweets')
    axs[0].set_title('Distribution of Tweet Lengths by Words')
    axs[0].grid(axis='y')
    axs[1].hist(tweet_lengths_chars, bins=50, align='left', color='salmon', edgecolor='black')
    axs[1].set_xlabel('Number of Characters')
    axs[1].set_ylabel('Number of Tweets')
    axs[1].set_title('Distribution of Tweet Lengths by Characters')
    axs[1].grid(axis='y')
    plt.tight_layout()
    plt.savefig("./tweet_length_distr")
    plt.close()  # Close the plot to release resources
    

    #plt.show()


    print(" The average length of the tweets is {} words ({} chars)".format(round(np.mean(tweet_lengths_words)),round(np.mean(tweet_lengths_chars))))

    # Sort the tweets based on retweet_count
    sorted_tweets = sorted(lines, key=lambda x: json.loads(x)['retweet_count'], reverse=True)
    for i, tweet in enumerate(sorted_tweets[:10]):
        print("{}.- {} \n\n(Retweets: {})\n\n".format(i+1, json.loads(tweet)['full_text'], json.loads(tweet)['retweet_count']))
    sorted_tweets = sorted(lines, key=lambda x: json.loads(x)['favorite_count'], reverse=True)


    for i, tweet in enumerate(sorted_tweets[:10]):
        print("{}.- {} \n\n(Favorite: {})\n\n".format(i+1, json.loads(tweet)['full_text'], json.loads(tweet)['favorite_count']))
        nlp = spacy.load("en_core_web_sm")



    # Perform entity recognition for each word in processed_lines
    entity_counts = defaultdict(int)

    for key in words.keys():
        entity = perform_entity_recognition(key)
        if entity != 0:
            if entity[0] in entity_counts:
                entity_counts[entity[0]] += 1
            else:
                entity_counts[entity[0]] = 1


    # Visualize entity type counts in a histogram
    plt.figure(figsize=(10, 6))
    plt.bar(entity_counts.keys(), entity_counts.values(), color='skyblue')
    plt.xticks(rotation=45)
    plt.title('Entity Type Counts in our Word Dict')
    plt.xlabel('Entity Type')
    plt.ylabel('Count')
    plt.savefig("./tweet_entity_distr")
    plt.close()  # Close the plot to release resources

    entity_counts = defaultdict(int)
    for line in lines:
        json_data = json.loads(line)
        entities = perform_entity_recognition(json_data['full_text'])
        if entities != 0:
            for entity in entities:
                if entity in entity_counts:
                    entity_counts[entity] += 1
                else:
                    entity_counts[entity] = 1


    # Visualize entity type counts in a histogram
    plt.figure(figsize=(10, 6))
    plt.bar(entity_counts.keys(), entity_counts.values(), color='skyblue')
    plt.xticks(rotation=45)
    plt.title('Entity Type Counts in the Tweets')
    plt.xlabel('Entity Type')
    plt.ylabel('Count')
    plt.savefig("./term_entity_distr")
    plt.close()  # Close the plot to release resources


if __name__ == '__main__':
    file_path = ''
    exploratory_analysis(read_tweets(file_path))