# Recommender-System-using-Multiple-Approaches

Built a Recommender System using multiple approaches.
The main file contains 3 functions, each a different approach. The 3 methods are:

- User Based Collaborative Filtering Recommender System 
- Singular Value Decomposition
- Alternating Least Squares factorization approach


User Based Collaborative Filtering:
It makes use of the similarity between the users to recommend movies and predicts movie ratings for a particular user who hasn't rated movies from the whole list.

Singular Value Decomposition:
Used the SVD Factorization approach to extract the relationship between the Users/Items and the latent factors as well as providing the strength of the relationship.
It provides the Singular values for the input matrix as well as the Left and Right Singular Matrices.

Alternating Least Squares:
Made use of PySpark in the cases of implementing the ALS Recommender System algorithm as it works well for large datasets.
It is a Collaborative Filtering approach and recommends movies to Users based on similar Users who watched the movies.

Datasets used:
Used the MovieLens dataset's Rating datset which contains the key features - UserId, MovieId, Timestamp and Rating.
Also used the MovieTitles dataset from MovieLens which contains the MovieName and Genre along with the MovieId
