### Developing e-commerce recommender system using the following techniques:

#### For collaborative filtering:
1) Matrix Factorization
2) NCF (Neural Collaborative Filtering)
3) RNN for Sequential Recommendation
4) Hybrid NCF + RNN
5) Graphs Network


#### Traditional statistical techniques:
1) Cosine similarity (btr than euclidean because it cares more about patterns rather than magnitude)
2) Euclidean Distance


#### For cold-start:
1) TF-IDF (looking at item's description and comparing)
2) Bag of Words
3) KNN
4) Word Embeddings (Word2Vec)


Some misc knowledge:<br>
Bayesian average to rank items with ratings + counts instead of just looking at their ratings.
Because items with 5 star ratings with only 2 ratings are not actually v popular vs an item with 4.5 star but 100+ ratings. 



#### In-depth matrix factorization

Supposed we have m users and n movies, 
If everyone were to have a rating for each movie, the total size would be m * n. 

We can use simple approach such as euclidean distance / cosine similarity to make the prediction. But, it is not scalable.
Because, if we have 500k by 500k matrix, the result matrix size is huge. And the simple approach require the full size of matrix to make the prediction. 

If we were to use matrix factorization instead, we are essentially decomposing the large matrix into 2 smaller ones. 
<BR>1 for user matrix
<BR>1 for item matrix

Both matrix are small because they are embedded to capture the latent factors that are most contributing to the outcome. 

Matrix factorization assumes that the rating 𝑟𝑢,𝑖 given by user 𝑢 to movie 𝑖 can be approximated by the dot product of the latent factor vectors for user 𝑢 and movie 𝑖:
ru,i ≈ Pu⋅Qi

Where:
𝑃𝑢 is the u-th row of the user matrix 𝑃.
𝑄𝑖 is the 𝑖-th row of the item matrix 𝑄.

This means the predicted rating is the dot product of the user's latent factor vector and the movie's latent factor vector.

We can adopt a simple SGD optimizer to compare the actual and predicted rating as well, with the loss function being RMSE. 
​

