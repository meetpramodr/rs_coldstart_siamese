import pandas as pd
import numpy as np
import argparse
from scipy.spatial.distance import cosine,euclidean

def get_distances(cold_movie,nearest_movies,n,i,dist='cosine'):
    cold_ratings = ratings.loc[ratings['movieId'] == imdbTomovieid[cold_movie],['userId','rating']]
    warm_movieslist = [imdbTomovieid[j.strip()] for j in nearest_movies.replace('[','').replace(']','').replace("'","").split(',')][:n]
    warmmovieToDist = {}
    for movie in warm_movieslist:
        warm_ratings = ratings.loc[ratings['movieId'] == movie,['userId','rating']]
        overlapping_userid = set(cold_ratings['userId']).intersection(set(warm_ratings['userId']))
        if len(overlapping_userid)>=50:
            vec_warm_iter = warm_ratings.loc[warm_ratings['userId'].isin(overlapping_userid)].sort_values(by='userId')
            vec_cold = cold_ratings.loc[cold_ratings['userId'].isin(overlapping_userid)].sort_values(by='userId')
            if dist=='cosine':
                distance = cosine(vec_cold['rating'],vec_warm_iter['rating'])
            elif dist == 'euclidean':
                distance = euclidean(vec_cold['rating'],vec_warm_iter['rating'])
            warmmovieToDist[movie] = distance
        else:
            warmmovieToDist[movie] = -1
    distlist = [k for k in list(warmmovieToDist.values()) if k!= -1]
    if len(distlist) > 0:
        mean_dist = np.mean(distlist)
    else:
        mean_dist = -99
    return mean_dist

def getndcg_at_N(df,n,i):
    cold_userids = ratings.loc[ratings['movieId'] == imdbTomovieid[df.loc[i,'cold_movie']],['userId','rating']]
    warm_movieslist = [imdbTomovieid[i.strip()] for i in df.loc[i,'closest_movies'].replace('[','').replace(']','').replace("'","").split(',')][:n]

    warm_userids_ratings = {}
    for j in warm_movieslist:
        userid_ratings = ratings.loc[ratings['movieId'] == j,['userId','rating']]
        for n,r in userid_ratings.iterrows():
            if r['userId'] not in warm_userids_ratings:
                warm_userids_ratings[r['userId']] = [r['rating']]
            else:
                warm_userids_ratings[r['userId']].append(r['rating'])
    warm_userids_ratings_avg = {x:y for x, y in sorted([(k,np.mean(v)) for k,v in warm_userids_ratings.items()] , key=lambda element: (element[1], element[0]), reverse=True)}

    predicted_userids = pd.DataFrame.from_dict(warm_userids_ratings_avg,orient='index',columns=['rating'])
    predicted_userids['userId'] = predicted_userids.index
    predicted_userids.reset_index(inplace=True,drop=True)
    predicted_userids.rename(columns={'rating':'predicted_rating'},inplace=True)

    actual_userids = cold_userids.sort_values(by=['rating','userId'], ascending=False).reset_index(drop=True)
    actual_userids.rename(columns={'rating':'actual_rating'},inplace=True)

    merged = pd.merge(left=actual_userids,right=predicted_userids,on='userId',how='inner')

    dcg = 0
    idcg = 0
    for n,row in merged.iterrows():
        dcg += (pow(2,row['predicted_rating']) - 1)/np.log2(n+2)
        idcg += (pow(2,row['actual_rating']) - 1)/np.log2(n+2)
    return (dcg,idcg)

def get_evaluation_mertrics(df,n):
    ndcg = []
    for j in range(len(df)):
        ndcg.append(getndcg_at_N(df,n,j))
        
    return(n,sum([x[0] for x in ndcg])/sum([x[1] for x in ndcg]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Results", 
                                    epilog="Usage Example: python evaluate_results.py "+
                                            "--siamese_file ../data/synopsis/siamese_pred_top100_googsyn_3feb3.csv "+
                                            "--features_file ../data/synopsis/w2vfeats_top100_googsyn_3feb3.csv ")
    parser.add_argument("--siamese_file",help="path of the siamese based result file",type=str)
    parser.add_argument("--features_file",help="path of the input features based result file",type=str)
    parser.add_argument("--ratings_file",help="path of the ratings.csv from ml-20m",type=str,default="../data/ml_20m/ratings.csv")
    parser.add_argument("--links_file",help="path of the links.csv from ml-20m",type=str,default="../data/ml_20m/links.csv")
    parser.add_argument("--n_neighbors",help="number of neighbors to be included (starts from 1 and goes upto n)",type=int,default=5)
    parser.add_argument("--distance_type",help="euclidean or cosine",type=str,default='cosine')

    cmd = parser.parse_args()
    args = dict()

    # Set input parameters
    args['siamese_file'] = cmd.siamese_file
    args['features_file'] = cmd.features_file
    args['ratings_file'] = cmd.ratings_file
    args['links_file'] = cmd.links_file
    args['n_neighbors'] = cmd.n_neighbors
    args['distance_type'] = cmd.distance_type

    # read files
    siamese_neighbors = pd.read_csv(cmd.siamese_file)
    features_neighbors = pd.read_csv(cmd.features_file)
    ratings = pd.read_csv(cmd.ratings_file)
    links = pd.read_csv(cmd.links_file)

    links['imdb_id'] = 'tt' + links['imdbId'].astype(str).str.zfill(7)
    movieidToimdb = dict(zip(links['movieId'].values,links['imdb_id'].values))
    imdbTomovieid = {v:k for k,v in movieidToimdb.items()}

    siamese_neighbors.sort_values(by='cold_movie',inplace=True)
    features_neighbors.sort_values(by='cold_movie',inplace=True)

    neighbors_num = range(1,cmd.n_neighbors)

    siamese_neighbornumTodist = {}
    w2v_neighbornumTodist = {}
    for n in neighbors_num:
        print("processing neighbor number:",n)
        
        #Computing Cosine Distances
        siamese_distlist = []
        w2v_distlist = []
        for i,row in siamese_neighbors.iterrows():
            siam_distance = get_distances(row['cold_movie'],row['closest_movies'],n,i,dist=cmd.distance_type)
            if siam_distance != -99:
                siamese_distlist.append(siam_distance)
        for j,row1 in features_neighbors.iterrows():
            w2v_distance = get_distances(row1['cold_movie'],row1['closest_movies'],n,j,dist=cmd.distance_type)
            if w2v_distance != -99:
                w2v_distlist.append(w2v_distance)
        siamese_neighbornumTodist[n] = np.mean(siamese_distlist)
        w2v_neighbornumTodist[n] = np.mean(w2v_distlist)
        
        #Computing NDCG
        grid_results_siamese ={}
        grid_results_feature_neighborhood ={}
        ndcg_results_siamese[n] = get_evaluation_mertrics(siamese_neighbors,n)
        ndcg_results_feature_neighborhood[n] = get_evaluation_mertrics(features_neighbors,n)
    
    print("feature_cosine:",w2v_neighbornumTodist)
    print("siamese_cosine:",siamese_neighbornumTodist)
    
    print("feature_ndcg:",ndcg_results_feature_neighborhood)
    print("siamese_ndcg:",ndcg_results_siamese)