import pandas as pd
import pickle
import numpy as np
from scripts.utils import *

if __name__ == "__main__":
    seed = 11
    cold_size = 333
    factors = 100

    #Fetch the ratings dataset
    ratings = read_ratings_data()

    #Filter ratings dataset to contain users who have givig min ratings and movies with min ratings
    ratings_filtered = filter_min_users_movies_from_ratings(ratings,min_users=50,min_movies=50)

    #Get the w2v features dictionary to subset the ratings
    with open('../data/synopsis/synopsis_imdbToFeatures.pkl','rb') as fp:
        imdbToW2VFeatures = pickle.load(fp)

    #Get the intersection of the ratings and w2v features
    intersection_df = ratings_filtered[ratings_filtered['imdb_id'].isin(list(imdbToW2VFeatures.keys()))]

    #Get warm and cold imdb ids
    unique_imdbIds = [movieIdToImdb[i] for i in intersection_df['movieId'].unique().tolist()]
    warm_imdbs, cold_imdbs = get_warm_cold_movies(unique_imdbIds,cold_size,seed,'../data/synopsis/cold_imdb.pkl')

    #Use only warm movies for collaborative filtering training data
    final_training_data = intersection_df.loc[intersection_df['movieId'].isin([imdbToMovieId[j] for j in warm_imdbs])]
    with open('../data/synopsis/synopsis_imdbToCollabFeatures.pkl','wb') as fp:
        pickle.dump(fit_collab(final_training_data,factors,userkey='userId',moviekey='imdb_id',rating='rating'),fp)






