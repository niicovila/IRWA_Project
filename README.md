# IRWA_Project 

The main function of the code is "evaluation", this function is the one in charge of processing the queries, both the baseline queries provided and the custom queries made by us. This function essentially creates a small index with the corresponding subset of documents (tweets) that belongs to each query test and then it computes the prior evaluation metrics with functions defined for every metric. 
To execute the evaluation of the baseline and custom queries it is only required to execute the “main()” function in “index.py” and it will print the metrics of evaluation for each query and then it will plot the 2D-scatter t-SNE of the word2vec representation of the tweets.
 

In evaluate_query.py, we have a functionality where the user can query our database of tweets and the system will write the ranked tweets in a txt file for the user to review. Essentially, it first creates the index for the entire tweets database, and has a loop where the user can provide queries.
