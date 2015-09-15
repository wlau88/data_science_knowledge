import numpy as np 
import scipy.sparse as ss
from time import time
import pdb


class MatrixFactorizationRec(object):


    def __init__(self,k,learning_rate,gamma):
        self.k = k
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.user_mat = None
        self.movie_mat = None
        self.n_users = None
        self.n_movies = None
        self.ratings_mat = None
        self.avg_movie_ratings = None
        self.user_bias = None

    def fit(self,ratings_mat,avg_movie_ratings,user_bias,n=100000,tol=0.001):
        
        # initialize information
        self.n_users = user_bias.shape[0]
        self.n_movies = avg_movie_ratings.shape[0]
        self.ratings_mat = ratings_mat
        self.user_bias = user_bias
        self.avg_movie_ratings = avg_movie_ratings
        self.user_mat = np.random.rand(self.n_users,self.k) * 0.1
        self.movie_mat = np.random.rand(self.k,self.n_movies) * 0.1
        num_ratings = ratings_mat.shape[0]

        # create first error metric
        error = 0
        pct_error = 100
        iterations = 0
        while pct_error > tol or iterations < 2e10:
            # calculate updates
            cur_update = np.random.randint(num_ratings)
            i = int(ratings_mat[cur_update,0]) - 1
            j = int(ratings_mat[cur_update,1]) - 1
            rating = ratings_mat[cur_update,2]
            row = self.user_mat[i,:]
            col = self.movie_mat[:,j]
            err = rating - (np.dot(row,col) + avg_movie_ratings[j] + user_bias[i])
            for k in range(self.k):
                uv = row[k]
                mv = col[k]
                self.user_mat[i,k] += self.learning_rate * (err * mv - self.gamma * uv)
                self.movie_mat[k,j] += self.learning_rate * (err * uv - self.gamma * mv)
            # check error every n loops
            if (iterations % n == 0):
                prev_error = error
                error = self.total_error()
                pct_error = 100 * float(prev_error-error)/error
                print iterations,',',error,',',pct_error
            iterations += 1

    # changed error metric to account for user and movie bias
    def total_error(self):
        error = 0
        for row in self.ratings_mat:
            mod = self.avg_movie_ratings[row[1]-1] + self.user_bias[row[0]-1]
            pred = self.user_mat[row[0]-1,:].dot(self.movie_mat[:,row[1]-1])
            error += (row[2] - (pred+mod))**2
        return error

    def pred_one_user(self,user_id):
        out = np.dot(self.user_mat[user_id-1],self.movie_mat) + self.avg_movie_ratings
        out += self.user_bias[user_id-1]
        return out

    def pred_all_users(self):
        out = np.dot(self.user_mat,self.movie_mat)
        return out

    def top_n_recs(self,user_id, n):
        pass
