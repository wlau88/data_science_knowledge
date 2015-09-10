import math
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.metrics.pairwise import cosine_similarity
from time import time
import cPickle
import pdb

from ItemRecommender import get_ratings_data, save_ratings_data, ItemItemRecommender

class MatrixFactorizationRecommender(ItemItemRecommender):
    """Basic framework for a matrix factorization class.

    You may find it useful to write some additional internal  methods for purposes
    of code cleanliness etc.
    """
    
    def __init__(self, n_features=8, learn_rate=0.005, regularization_param=0.02,
                 optimizer_pct_improvement_criterion=2):
        """Init should set all of the parameters of the model
        so that they can be used in other methods.
        """

        self.user_mat = None
        self.movie_mat = None
        self.rating_matrix = None
        self.n_features = n_features
        self.learn_rate = learn_rate
        self.regularization_param = regularization_param
        self.optimizer_pct_improvement_criterion = optimizer_pct_improvement_criterion

    def fit(self, ratings_mat):
        """Like the scikit learn fit methods, this method 
        should take the ratings data as an input and should
        compute and store the matrix factorization. It should assign
        some class variables like n_users, which depend on the
        ratings_mat data.

        It can return nothing
        """
        self.rating_matrix = ratings_mat
        n_users = ratings_mat.shape[0]
        n_movies = ratings_mat.shape[1]
        n_already_rated = ratings_mat.nonzero()[0].size
        user_mat = np.random.rand(
            n_users*self.n_features).reshape([n_users, self.n_features])
        movie_mat = np.random.rand(
            n_movies*self.n_features).reshape([self.n_features, n_movies])

        optimizer_iteration_count = 0
        sse_accum = 0
        print("Optimizaiton Statistics")
        print("Iterations | Mean Squared Error | Percent Improvement")
        while (optimizer_iteration_count < 2 or (pct_improvement > 
            self.optimizer_pct_improvement_criterion)):
            old_sse = sse_accum
            sse_accum = 0
            for i in xrange(n_users):
                for j in xrange(n_movies):
                    if ratings_mat[i, j] > 0:
                        error = ratings_mat[i, j] - \
                            np.dot(user_mat[i, :], movie_mat[:, j])
                        sse_accum += error**2
                        for k in xrange(self.n_features):
                            delta_u = self.learn_rate * (2 * error * movie_mat[k, j] - self.regularization_param * user_mat[i, k])
                            user_mat[i, k] = user_mat[i, k] + delta_u 
                            
                            delta_v = self.learn_rate * (2 * error * user_mat[i, k] - self.regularization_param * movie_mat[k, j])
                            movie_mat[k, j] = movie_mat[k, j] + delta_v
            pct_improvement = 100 * (old_sse - sse_accum) / old_sse
            print("%d \t\t %f \t\t %f" % (
                optimizer_iteration_count, sse_accum / n_already_rated, pct_improvement))
            old_sse = sse_accum
            optimizer_iteration_count += 1
        # ensure these are matrices so multiplication works as intended
        
        self.user_mat = np.matrix(user_mat)
        self.movie_mat = np.matrix(movie_mat)
    
    def pred_one_user(self, user_id):
        """Returns the predicted rating for a single
        user.
        """
        
        return ( np.asarray( self.user_mat[user_id] * self.movie_mat) )[0]
    
    def top_n_recs(self, user_id, num):
        """Returns the top n recs for a given user.
        """

        user_rating_TF = self.rating_matrix.todense()[user_id, :] > 0 
        user_pred = self.pred_one_user(user_id)
        user_pred_minus_user_rating = np.where(user_rating_TF, 0, user_pred)

        return np.argsort(-user_pred_minus_user_rating)[ 0,  : num ]


def validation(recommendation_obj, user_pct, item_pct, rating_matrix, score_function):
    n_users = rating_matrix.shape[0]
    n_items = rating_matrix.shape[1]
    oos_users = int(user_pct * n_users)
    oos_items = int(item_pct * n_items)

    out_of_sample = rating_matrix[:oos_users, :oos_items]
    training_matrix = rating_matrix.copy()
    training_matrix[:oos_users, :oos_items] = np.zeros_like(out_of_sample)

    recommendation_obj.fit( training_matrix )
    pred_matrix = recommendation_obj.pred_all_users()

    pred_out_of_sample = pred_matrix[:oos_users, :oos_items]

    return score_function(out_of_sample, pred_out_of_sample)

def mse_sparse_with_dense(sparse_mat, dense_mat):
    """
    Computes mean-squared-error between a sparse and a dense matrix.  
    Does not include the 0's from the sparse matrix in computation (treats them as missing)
    """
    #get mask of non-zero, mean-square of those, divide by count of those
    nonzero_idx = sparse_mat.nonzero()
    mse = (np.array(sparse_mat[nonzero_idx] - dense_mat[nonzero_idx])**2).mean()
    return mse

def get_item():
    item_names = ["movie id", "movie title", "release date", 
    "video release date", "IMDb URL", "unknown", "Action", "Adventure", 
    "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", 
    "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", 
    "Thriller", "War", "Western"]
    return pd.read_table("../data/u.item", sep="|", header=None, names=item_names)

if __name__ == "__main__":

    contents = pd.read_pickle('../data/df_contents.pkl')
    with open('../data/sparse_mat.pkl','r') as f:
        mat = cPickle.load( f )

    ii_rec = ItemItemRecommender()
    ii_rec.fit(mat)

    ## Run once
    # mfr_rec = MatrixFactorizationRecommender()
    # mfr_rec.fit(mat)

    # with open('../data/MFR.pkl','w') as f:
    #     cPickle.dump( mfr_rec, f, -1 )
    # assert False

    with open('../data/MFR.pkl','r') as f:
        mfr_rec = cPickle.load( f )

    print mfr_rec.pred_one_user(1)
    print mfr_rec.pred_all_users(run_time=True).shape
    movie_recommendations = mfr_rec.top_n_recs(1, 10)

    recommendation_obj = mfr_rec
    user_pct, item_pct = .1, .1
    rating_matrix = mat
    score_function = mse_sparse_with_dense

    # print validation(recommendation_obj, user_pct, item_pct, rating_matrix, score_function)

    movies_df = get_item()
    print movies_df['movie title'][movie_recommendations]
    ratings_series = np.asarray(mat[1].todense())[0][ np.asarray( mat[1].todense().nonzero()[1] )[0] ]
    # print mat[1][mat[1].nonzero()[1]].todense().shape

    print pd.DataFrame({'title':(movies_df['movie title'][mat[1].nonzero()[1]]), 
        'rating':ratings_series}).sort('rating', ascending=False)

'''
A couple of notes:
The predicted ratings are mostly between 2 and 4 in UV Decomposition, as opposed to 
4.5 in the ItemItemRecommender because the UV matrix factorization method takes into 
account both a user's rating and a movie's rating, i.e. more of an average which will 
restrict extreme values like 4.5 in the ItemItem case.

The validation strategy as implemented favors the ItemItem case because some of the elements 
in the neighborhood might be the original values. While all of the values in UV are new 
values created from the decomposition.
'''