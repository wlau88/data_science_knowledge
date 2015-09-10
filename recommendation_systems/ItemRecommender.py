import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from time import time
import cPickle


class ItemItemRecommender(object):
    def __init__(self, neighborhood_size=75):
        """Initializes the parameters of the model.
        """
        self.neighborhood_size = neighborhood_size
        self.items_cos_sim = None
        self.neighborhoods = None
        self.rating_matrix = None


    def fit(self, rating_matrix):
        """Implements the model and fits it to the data passed as an
        argument.
        Stores objects for describing model fit as class attributes.
        """
        self.rating_matrix = rating_matrix
        self._set_neighborhoods()
    

    def _set_neighborhoods(self):
        """Gets the items most similar to each other item.
        Should set a class attribute with a matrix that is has
        number of rows equal to number of items and number of 
        columns equal to neighborhood size. Entries of this matrix
        will be indexes of other items.
        You will call this in your fit method.
        """
        self.items_cos_sim = cosine_similarity(self.rating_matrix.T)
        least_to_most_sim_indexes = np.argsort(self.items_cos_sim, 1)
        self.neighborhoods = least_to_most_sim_indexes[:, -self.neighborhood_size:]


    def pred_one_user(self, user_id, run_time=False, all_pred=None):
        """Accept user id as arg. Return the predictions for a single user.
        
        Optional argument to specify whether or not timing should be provided
        on this operation.
        """
        start = time()

        if all_pred is not None:
            if run_time:
                print "Run time (fast):", time() - start
            return all_pred[user_id,:]
        
        n_items = self.rating_matrix.shape[1]
        items_rated_by_this_user = self.rating_matrix[user_id].nonzero()[1]
        # Just initializing so we have somewhere to put rating preds
        output = np.zeros(n_items)

        for item_to_rate in xrange(n_items):
            relevant_items = np.intersect1d(self.neighborhoods[item_to_rate],
                                            items_rated_by_this_user,
                                            assume_unique=True)  # assume_unique speeds up intersection op
            output[item_to_rate] = self.rating_matrix[user_id, relevant_items] * \
                self.items_cos_sim[item_to_rate, relevant_items] / \
                self.items_cos_sim[item_to_rate, relevant_items].sum()

        if run_time:
            print "Run time:", time() - start
        return np.nan_to_num( output )
        

    def pred_all_users(self, run_time=False, num_test=None):
        """Repeated calls of pred_one_user, are combined into a single matrix.
        Return value is matrix of users (rows) items (columns) and predicted
        ratings (values).
        Optional argument to specify whether or not timing should be provided
        on this operation.
        """
        start = time()

        nrows = self.rating_matrix.shape[0]
        if num_test:
            nrows = num_test

        pred_list = []
        for user in xrange(nrows):
            pred_list.append(self.pred_one_user(user))

        if run_time:
            print "Run time:", time() - start
        return np.asarray(pred_list)


    def top_n_recs(self, user_id, num, all_pred=None):
        """Takes user_id argument and number argument.
        Returns that number of items with the highest predicted ratings,
        after removing items that user has already rated.
        """
    
        user_rating_TF = self.rating_matrix.todense()[user_id,:] > 0 
        user_pred = self.pred_one_user(user_id, all_pred=all_pred)
        user_pred_minus_user_rating = np.where(user_rating_TF, 0, user_pred)

        return np.argsort(-user_pred_minus_user_rating)[0, :num]


def get_ratings_data():
    ratings_contents = pd.read_table("../data/u.data",
                                     names=["user", "movie", "rating", "timestamp"])
    highest_user_id = ratings_contents.user.max()
    highest_movie_id = ratings_contents.movie.max()
    ratings_as_mat = sparse.lil_matrix((highest_user_id, highest_movie_id))
    for _, row in ratings_contents.iterrows():
        # subtract 1 from id's due to match 0 indexing
        ratings_as_mat[row.user-1, row.movie-1] = row.rating
    return ratings_contents, ratings_as_mat


def save_ratings_data():
    contents, mat = get_ratings_data()

    contents.to_pickle('../data/df_contents.pkl')
    with open('../data/sparse_mat.pkl','w') as f:
        cPickle.dump( mat, f, -1 )


if __name__ == "__main__":

    ## Run once
    # save_ratings_data()
    # assert False

    contents = pd.read_pickle('../data/df_contents.pkl')
    with open('../data/sparse_mat.pkl','r') as f:
        mat = cPickle.load( f )

    ii_rec = ItemItemRecommender()
    ii_rec.fit(mat)

    ## Each user's prediction takes ~0.57 seconds, thus it will take
    ## (0.571 * 943) / 60. ~ 9 minutes to predict all users
    
    ## Run once
    # all_users = ii_rec.pred_all_users(run_time=True)
    # all_users.dump('../data/all_user_pred.pkl')

    all_users = np.load('../data/all_user_pred.pkl')
    print all_users.shape

    print ii_rec.pred_one_user(1, run_time=True)
    print ii_rec.pred_one_user(1, True, all_users)

    ## Calculating each prediction in real time takes 0.56 seconds, while looking up
    ## the precomputed array is only 7e-06.

    ## The downside is we have to compute and store a very large matrix, and update (likely)
    ## several times a day. Hence recommendation won't be updated in real time.

    print ii_rec.top_n_recs(1, 10, all_users)