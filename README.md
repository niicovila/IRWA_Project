# IRWA_Project 

**Part2**:

- The main function of the code is "evaluation", this function is the one in charge of processing the queries, both the baseline queries provided and the custom queries made by us. This function essentially creates a small index with the corresponding subset of documents (tweets) that belongs to each query test and then it computes the prior evaluation metrics with functions defined for every metric. 
To execute the evaluation of the baseline and custom queries it is only required to execute the “main()” function in “index.py” and it will print the metrics of evaluation for each query and then it will plot the 2D-scatter t-SNE of the word2vec representation of the tweets.
 

- In evaluate_query.py, we have a functionality where the user can query our database of tweets and the system will write the ranked tweets in a txt file for the user to review. Essentially, it first creates the index for the entire tweets database, and has a loop where the user can provide queries.

**Part3**:

- Ranking Scores: Once you execute the 'execute_query.py', the program will take a few minutes creating the index and the embeddings we created for the tweets. After that, it will ask the user for a query, which will be used to rank the tweets according to two different ranking methods. The first one is the traditional tf-idf score with cosine similarity, as already implemented in the last part. The second score, from our choice, consists on a function of the cosine similarity of the tweets and query embeddings and the likes and retweets of the tweets. Thus, two files will be created to store the results: ranking_tfidf.txt will store the top 20 tweets according the tf-idf score, and ranking_custom.txt will store the top 20 tweets according our custom score. The k value is set by default at 20, since it is what the statement asked for, but it can be easily changed in the code.

- Word2vec: The implementation of our 'Word2Vec' ranking, or rather 'Tweet2Vec' works as following. We first compute a data structure that stores the tweet vector for each tweetId, in that way, we don't need to recompute the vectors for each query. Secondly, there is a loop that asks the user for an input. For each given input, we obtain the terms of the query that are in our index, and retrieve only those documents that contain all the terms. Subsequently, we retrieve a subset of our vector data structure to retrieve only those tweets that matched all terms in the query, and perform cosine similarity only with that subset. Finally, we write the top 20 results in a text file. Between all of our 5 queries, we only got matching tweets for the first one, since it was the only one where all the query terms - present in our index- were also in the tweet that was retrieved. This makes sense because in our case, our queries are a bit large for this rather simple model of retrieving tweets. When a query is too long, it gets increasingly more rare that the model will return any matching query. That is because of what I just mentioned, which is that we only retrieve tweets that have all the terms of the query -present in the index-.


**Part4**:

## Starting the Web App

```bash
python -V
# Make sure we use Python 3

cd search-engine-web-app
python web_app.py
```
The above will start a web server with the application:
```
 * Serving Flask app 'web-app' (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://127.0.0.1:8088/ (Press CTRL+C to quit)
```

Open Web app in your Browser:  
[http://127.0.0.1:8088/](http://127.0.0.1:8088/) or [http://localhost:8088/](http://localhost:8088/)

- The UI of the search engine was crafted to provide a simple and efficient user experience. The search interface was designed with clarity in mind, featuring a straightforward layout with a prominent search bar, enabling users to enter their queries with minimal distraction. This design choice was motivated by the need to create an environment where users could focus on their search tasks without unnecessary complexity or clutter.

- In displaying search results, attention was paid to both aesthetics and functionality. Each result was presented in a clean and organized manner, making it easy for users to scan through the information and identify relevant results. The use of well-defined sections, clear typography, and spacing contributed to the readability and overall user-friendliness of the results page. This approach not only enhanced the visual appeal of the application but also improved usability, a key factor in user satisfaction.

- Clickable elements in the search results were designed to be intuitive, encouraging user interaction while simultaneously facilitating the collection of valuable click data. This feature was subtly incorporated into the UI, ensuring that it did not disrupt the user experience or detract from the primary functionality of the search engine.

- For data collection and storage, an in-memory storage mechanism was chosen. This decision was made to simplify the replication process bypassing the complexities associated with setting up and managing a database. Custom data models for Sessions, Clicks, and Requests were developed within the Python and Flask framework, reflecting a preference for a straightforward and accessible development environment conducive to rapid prototyping. While this approach was ideal for demonstration and educational purposes, it was recognized that a persistent database solution would be more appropriate for real-world applications where data durability and scalability are crucial.

  ## Importance of Chosen Metrics

  1. Session Data (User IP, Location, User Agent): Collecting session data is fundamental for understanding user demographics and behavior. Information like user IP, geographical location, and user agent provides insights into the diversity of the user base, their geographical distribution, and the devices or browsers used. This data is crucial for optimizing the application for different user segments and ensuring compatibility across various platforms.

2. Request Data (Queries, Timestamps): Tracking search queries and their timestamps enables the analysis of search trends, popular search terms, and user engagement over time. Understanding what users are searching for is vital for refining the search algorithm and ensuring that it aligns with user needs and expectations. Timestamp data helps in analyzing peak usage times, which can inform infrastructure scaling decisions and maintenance scheduling.

3. Click Data (Document Clicks, Related Queries, Ranking): Monitoring which documents are clicked, the associated queries, and their rankings in the search results yields valuable insights into the effectiveness of the search algorithm. It helps in assessing whether the most relevant and useful results are being presented to the users. This data is also instrumental in improving search result ranking algorithms.



