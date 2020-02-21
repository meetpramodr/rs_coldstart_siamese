#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path = ["/mnt/home/my_lib/"] + sys.path

import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pickle

from tqdm.notebook import tqdm


# In[2]:


import seaborn as sns
sns.set()


# ## Read Data

# In[3]:


imdbidToCollabFeatures_meta = pickle.load(open('./imdbidToCollabFeatures_trailer_11feb.pkl','rb'))

movies = []
collab_feats = []
for k,v in imdbidToCollabFeatures_meta.items():
    movies.append(k)
    collab_feats.append(v)
collab_df = pd.DataFrame()
collab_df['movies'] = movies
collab_df['feats'] = collab_feats
imdbidToCollabFeatures_meta = ''
movies = []
collab_feats = []


# In[4]:


trailer_feats1 = pd.read_csv('./merged_alexnet_avg1.csv',encoding = "ISO-8859-1")
trailer_feats2 = pd.read_csv('./merged_alexnet_avg2.csv',encoding = "ISO-8859-1")
trailer_feats3 = pd.read_csv('./merged_alexnet_avg3.csv',encoding = "ISO-8859-1")


# In[5]:


trailer_feats = pd.concat([trailer_feats1,trailer_feats2,trailer_feats3],axis=0).reset_index(drop=True)


# In[6]:


movieIdToImdbid = pickle.load(open('./movieIdToImdbid.pkl','rb'))


# In[7]:


trailer_feats['movie_id'] = [movieIdToImdbid[x] for x in trailer_feats['movieId']]


# In[8]:


trailer_feats_all = pd.DataFrame()
trailer_feats_all['imdb_id'] = trailer_feats['movie_id']
trailer_feats_all['feats'] = list(trailer_feats[list(trailer_feats.columns)[3:-1]].values)


# In[9]:


# feat_cols = list(movie_features.columns)[2:]
movie_trailer_df = pd.DataFrame()
movie_trailer_df['movies'] = trailer_feats_all['imdb_id']
movie_trailer_df['feats'] = trailer_feats_all['feats']


# In[10]:


from sklearn.preprocessing import StandardScaler


# In[11]:


sc = StandardScaler(with_std=True)
movie_trailer_df['scaled_feats'] = list(sc.fit_transform(list(movie_trailer_df['feats'])))
pickle.dump(sc,open('scaler_trailer.pkl','wb'))


# In[12]:


movie_trailer_dict = {}
for ind,row in movie_trailer_df.iterrows():
    movie_trailer_dict[row['movies']] = row['scaled_feats']


# In[13]:


len(set(movie_trailer_df['movies']).intersection(collab_df['movies']))


# In[14]:


cold_movies = pickle.load(open('./cold_movies_trailers_11feb.pkl','rb'))
common_cold_movies = list(set(cold_movies).intersection(set(movie_trailer_df['movies'])))
len(common_cold_movies)


# In[15]:


featsize = movie_trailer_df['scaled_feats'][0].shape[0]
featsize


# ## Network

# In[16]:


import keras

from keras.models import Sequential
import time
from keras.optimizers import Adam
from keras.layers import Activation, Input, concatenate, Dropout
from keras.models import Model
import seaborn as sns
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import *
from keras.engine.topology import Layer
from keras import backend as K
from keras.regularizers import l2
from keras.utils import plot_model


# In[17]:


from keras import regularizers
from keras.layers import Dot


# In[18]:


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


# In[19]:


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


# In[20]:


input_shape = (featsize,)
left_input = Input(input_shape)
right_input = Input(input_shape)
model = Sequential()
reg_param = 0.000000005
model.add(Dense(4096, activation='tanh', input_shape=input_shape,
                kernel_regularizer=regularizers.l2(reg_param),
                activity_regularizer=regularizers.l1(reg_param),kernel_initializer='random_uniform',
                bias_initializer='zeros'))
model.add(Dropout(0.05))
model.add(Dense(2048, activation='tanh',
                kernel_regularizer=regularizers.l2(reg_param),
                activity_regularizer=regularizers.l1(reg_param),kernel_initializer='random_uniform',
                bias_initializer='zeros'))
model.add(Dropout(0.05))
model.add(Dense(1024, activation='tanh',
                kernel_regularizer=regularizers.l2(reg_param),
                activity_regularizer=regularizers.l1(reg_param),kernel_initializer='random_uniform',
                bias_initializer='zeros'))
model.add(Dropout(0.05))
model.add(Dense(256, activation='tanh', input_shape=input_shape,
                kernel_regularizer=regularizers.l2(reg_param),
                activity_regularizer=regularizers.l1(reg_param),kernel_initializer='random_uniform',
                bias_initializer='zeros'))

# model.add(Dropout(0.10 ))


encoded_l = model(left_input)
encoded_r = model(right_input)




L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
L1_distance = L1_layer([encoded_l, encoded_r])
L1_distance_tan = Dense(1,activation='tanh')(L1_distance)
prediction = Dense(1,activation='linear')(L1_distance_tan)
neg_layer = Lambda(lambda tensor:-1*tensor)
neg_prediction = Dense(1)(neg_layer(L1_distance_tan))
siamese_net = Model(inputs=[left_input,right_input],outputs=neg_prediction)


optimizer = Adam(lr = 0.00005)
siamese_net.compile(loss='mse',optimizer=optimizer)

siamese_net.summary()


# ## Training data make

# In[21]:


from scipy.spatial.distance import cdist


# In[22]:


sc1 = StandardScaler()
all_collab_feats = list(sc1.fit_transform(np.array([np.array(x) for x in collab_df['feats']])))


# In[23]:


len(all_collab_feats)


# In[24]:


all_dists = 1 - cdist(all_collab_feats,all_collab_feats,metric = 'cosine')


# In[25]:


training_dists = []
all_movies = list(collab_df['movies'])
i=0
for movie1 in tqdm(all_movies):
    for j,movie2 in enumerate(all_movies):
        if j<=i or movie_trailer_dict.get(movie1) is None or movie_trailer_dict.get(movie2) is None:
            continue
        else:
            training_dists.append(((movie1,movie2),all_dists[i,j]))
    i+=1


# In[26]:


training_inputs1 = []
training_inputs2 = []
training_outputs = []
for item in tqdm(training_dists):
    training_inputs1.append(movie_trailer_dict[item[0][0]])
    training_inputs2.append(movie_trailer_dict[item[0][1]])
    training_outputs.append(item[1])


# In[27]:


training_df = pd.DataFrame()
training_df['X1'] = training_inputs1
training_df['X2'] = training_inputs2
training_df['Y'] = training_outputs


# ## Test-Train Split

# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


train_df,test_df = train_test_split(training_df, test_size=0.2, random_state=42)


# In[33]:


# training_inputs = ''
# training_outputs = ''


# ## Training

# In[34]:


step = 300000
# train_df = training_df
losses = []
for epoch in tqdm(range(100),desc = 'Epochs'):
    loss = []
    for i in tqdm(range(0,train_df.shape[0],step),desc = 'steps'):
        X_train1 = np.array(list(train_df['X1'].values)[i:i+step])
        X_train2 = np.array(list(train_df['X2'].values)[i:i+step])
        y_train = list(train_df['Y'])[i:i+step]
        siamese_net.fit([X_train1,X_train2],y_train,batch_size=2048,validation_split=0.2,epochs=1,)
        loss.append(siamese_net.history.history)
    losses.append(loss)


# In[52]:


siamese_net.save('siamese_trailer_model4.model')


# In[59]:


plt.figure(figsize=(16,10))
plt.plot([x['loss'] for x in losses2[-1]],'-o',label='training')
plt.plot([x['val_loss'] for x in losses2[-1]],'-o',label='validation')
plt.title('Training loss')
plt.ylabel('Loss')
plt.xlabel('Iterations-->>')
plt.legend()
plt.show()


# ## Cold movies

# In[55]:


cold_movies = pickle.load(open('./cold_movies_trailers_11feb.pkl','rb'))
common_cold_movies = list(set(cold_movies).intersection(set(movie_trailer_df['movies'])))
len(common_cold_movies)


# In[56]:


# movie_trailer_dict = pickle.load(open('movie_trailer_dict.pkl','rb'))

total_warm_movies = collab_df['movies'].shape[0]
all_warm_movies = list(collab_df['movies'])

closest_movies_list = []
input1 = np.array([movie_trailer_dict[warm_movie] for warm_movie in all_warm_movies])
for cold_movie in tqdm(common_cold_movies):
    input2 = np.repeat(movie_trailer_dict[cold_movie].reshape(1,featsize),total_warm_movies,axis =0)
    preds = siamese_net.predict([input1,input2])
    closest_movies = [all_warm_movies[x] for x in list(np.argsort(-1*np.array([a[0] for a in list(preds)]))[:100])]
    closest_movies_list.append(closest_movies)

closest_movies_df = pd.DataFrame()
closest_movies_df['cold_movie'] = common_cold_movies
closest_movies_df['closest_movies'] = closest_movies_list

movieIdToImdbId = pd.read_pickle('movieIdToImdbid.pkl')

movies = pd.read_csv('movies.csv')
movies['imdbId'] = [movieIdToImdbId[x] for x in movies['movieId']]
imdbIdToMovie = {}
for k,row in movies.iterrows():
    imdbIdToMovie[row['imdbId']] = row['title']

cold_movie_names = []
closest_movie_names = []
for k,row in closest_movies_df.iterrows():
    cold_movie_names.append(imdbIdToMovie[row['cold_movie']])
    closest_movieIds = row['closest_movies']
    closest_movies = [imdbIdToMovie[x] for x in closest_movieIds]
    closest_movie_names.append(closest_movies)
closest_movies_df['cold_movie_names'] = cold_movie_names
closest_movies_df['closest_movie_names'] = closest_movie_names

closest_movies_df.to_csv('cold_movies_trailer_by_network4.csv',index = False)


# In[57]:


pd.read_csv('cold_movies_trailer_by_network4.csv').head(60)


# In[58]:


movie_meta_dict = movie_trailer_dict



closest_movies_list = []
input1 = np.array([movie_meta_dict[warm_movie] for warm_movie in all_warm_movies])
for cold_movie in tqdm(common_cold_movies):
    input2 = movie_meta_dict[cold_movie]
    preds = 1 - cdist(input2.reshape((1,featsize)),input1)
    closest_movies = [all_warm_movies[x] for x in list(np.argsort(-1*np.array([a for a in list(preds[0])]))[:100])]
    closest_movies_list.append(closest_movies)

from scipy.spatial.distance import cdist

closest_movies_df = pd.DataFrame()
closest_movies_df['cold_movie'] = common_cold_movies
closest_movies_df['closest_movies'] = closest_movies_list

movies = pd.read_csv('movies.csv')
movies['imdbId'] = [movieIdToImdbId[x] for x in movies['movieId']]
imdbIdToMovie = {}
for k,row in movies.iterrows():
    imdbIdToMovie[row['imdbId']] = row['title']

cold_movie_names = []
closest_movie_names = []
for k,row in closest_movies_df.iterrows():
    cold_movie_names.append(imdbIdToMovie[row['cold_movie']])
    closest_movieIds = row['closest_movies']
    closest_movies = [imdbIdToMovie[x] for x in closest_movieIds]
    closest_movie_names.append(closest_movies)
closest_movies_df['cold_movie_names'] = cold_movie_names
closest_movies_df['closest_movie_names'] = closest_movie_names

closest_movies_df.to_csv('100closest_by_trailer_by_dist4.csv',index = False)


# In[ ]:




