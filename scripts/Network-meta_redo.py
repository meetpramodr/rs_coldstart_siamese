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


# ## Read Data

# In[2]:


imdbidToCollabFeatures_meta = pickle.load(open('./imdbidToCollabFeatures_meta_11feb.pkl','rb'))

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


# In[3]:


movie_features = pd.read_pickle('movie_features2.pkl')

feat_cols = list(movie_features.columns)[2:]
movie_meta_df = pd.DataFrame()
movie_meta_df['movies'] = movie_features['new_tconst']
movie_meta_df['feats'] = list(movie_features[feat_cols].values)
movie_features = ''
feat_cols = ''


# In[4]:


feats_float = []
for ind,row in movie_meta_df.iterrows():
    feats_float.append(np.array([float(x) if str(x).isnumeric() else 0 for x in row['feats']]))
movie_meta_df['feats_float'] = feats_float


# In[5]:


from sklearn.preprocessing import StandardScaler


# In[8]:


sc = StandardScaler()
movie_meta_df['scaled_feats'] = list(sc.fit_transform(feats_float))
pickle.dump(sc,open('scaler.pkl','wb'))


# In[9]:


movie_meta_dict = {}
for ind,row in movie_meta_df.iterrows():
    movie_meta_dict[row['movies']] = row['scaled_feats']


# In[10]:


len(set(movie_meta_df['movies']).intersection(collab_df['movies']))


# In[11]:


featsize = movie_meta_df['feats'].values[0].shape[0]


# ## Network

# In[12]:


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


# In[13]:


from keras import regularizers


# In[14]:


input_shape = (featsize,)
left_input = Input(input_shape)
right_input = Input(input_shape)
model = Sequential()
reg_param = 0.000000001
model.add(Dense(1024, activation='tanh', input_shape=input_shape,kernel_regularizer=regularizers.l2(reg_param),
                activity_regularizer=regularizers.l1(reg_param),kernel_initializer='random_uniform',
                bias_initializer='zeros'))
model.add(Dropout(0.05))
model.add(Dense(512, activation='tanh',kernel_regularizer=regularizers.l2(reg_param),
                activity_regularizer=regularizers.l1(reg_param),kernel_initializer='random_uniform',
                bias_initializer='zeros'))
model.add(Dropout(0.05))
model.add(Dense(100, activation='tanh',kernel_regularizer=regularizers.l2(reg_param),
                activity_regularizer=regularizers.l1(reg_param),kernel_initializer='random_uniform',
                bias_initializer='zeros'))
model.add(Dropout(0.0))

encoded_l = model(left_input)
encoded_r = model(right_input)


######### L1 distance ################
L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
L1_distance = L1_layer([encoded_l, encoded_r])
L1_distance_tan = Dense(1,activation='tanh')(L1_distance)
prediction = Dense(1,activation='linear')(L1_distance_tan)
neg_layer = Lambda(lambda tensor:-1*tensor)
neg_prediction = Dense(1)(neg_layer(prediction))
siamese_net = Model(inputs=[left_input,right_input],outputs=neg_prediction)



optimizer = Adam(lr = 0.0005)
siamese_net.compile(loss='mse',optimizer=optimizer)

siamese_net.summary()


# ## Training data make

# In[15]:


from scipy.spatial.distance import cdist


# In[16]:


sc1 = StandardScaler(with_mean=False, with_std=False)
all_collab_feats = sc1.fit_transform(np.array([np.array(x) for x in collab_df['feats']]))


# In[18]:


all_dists = 1 - cdist(all_collab_feats,all_collab_feats,metric='cosine')


# In[20]:


training_dists = []
all_movies = list(collab_df['movies'])
i=0
for movie1 in tqdm(all_movies):
    for j,movie2 in enumerate(all_movies):
        if j<=i or np.random.random()>0.7:
            continue
        else:
            training_dists.append(((movie1,movie2),all_dists[i,j]))
    i+=1


# In[21]:


training_inputs1 = []
training_inputs2 = []
training_outputs = []
for item in tqdm(training_dists):
    training_inputs1.append(movie_meta_dict[item[0][0]])
    training_inputs2.append(movie_meta_dict[item[0][1]])
    training_outputs.append(item[1])


# In[22]:


training_df = pd.DataFrame()
training_df['X1'] = training_inputs1
training_df['X2'] = training_inputs2
training_df['Y'] = training_outputs


# ## Test-Train Split

# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


train_df,test_df = train_test_split(training_df, test_size=0.2, random_state=42)


# In[29]:


training_inputs = ''
training_outputs = ''


# ## Training

# In[30]:


step = 1000000


# In[31]:


losses = []
for epoch in tqdm(range(100),desc = 'Epochs'):
    loss = []
    for i in tqdm(range(0,train_df.shape[0],step),desc = 'steps'):
        X_train1 = np.array(list(train_df['X1'].values)[i:i+step])
        X_train2 = np.array(list(train_df['X2'].values)[i:i+step])
        y_train = list(train_df['Y'])[i:i+step]
        siamese_net.fit([X_train1,X_train2],y_train,batch_size=2048,validation_split=0.1,epochs=1,)
        loss.append(siamese_net.history.history)
    losses.append(loss)


# In[32]:


siamese_net.save('siamese_meta_redo_11thFeb2.h5')
pickle.dump(losses,open('losses_meta_redo_11thFeb2.pkl','wb'))


# In[33]:


all_loss_training = []
all_loss_val = []
for l in losses:
    for y in l:
        all_loss_training.append(y['loss'])
        all_loss_val.append(y['val_loss'])
        
plt.figure(figsize=(12,8))
plt.plot(all_loss_training[1:],'-o',label='Training')
plt.ylabel('Loss')

plt.plot(all_loss_val[1:],'-o',label='Validation')
plt.xlabel('iter')

plt.title('Metadata Training Validation redo')
plt.legend()
plt.show()


# # Cold movies

# In[34]:


cold_movies = pickle.load(open('./cold_movies_imdb_11feb.pkl','rb'))
common_cold_movies = list(set(cold_movies).intersection(set(movie_meta_df['movies'])))
len(common_cold_movies)


# In[35]:


# movie_meta_dict = pickle.load(open('movie_meta_dict.pkl','rb'))

total_warm_movies = collab_df['movies'].shape[0]
all_warm_movies = list(collab_df['movies'])

closest_movies_list = []
input1 = np.array([movie_meta_dict[warm_movie] for warm_movie in all_warm_movies])
for cold_movie in tqdm(common_cold_movies):
    input2 = np.repeat(movie_meta_dict[cold_movie].reshape(1,featsize),total_warm_movies,axis =0)
    preds = siamese_net.predict([input1,input2])
    closest_movies = [all_warm_movies[x] for x in list(-1*np.argsort(np.array([a[0] for a in list(preds)]))[:100])]
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

closest_movies_df.to_csv('closest_siamese_meta_redo2.csv',index = False)



# In[36]:


closest_movies_df[['cold_movie_names','closest_movie_names']].head(60)


# In[37]:


closest_movies_list = []
input1 = np.array([movie_meta_dict[warm_movie] for warm_movie in all_warm_movies])
for cold_movie in tqdm(common_cold_movies):
    input2 = movie_meta_dict[cold_movie]
    preds = 1 - cdist(input2.reshape((1,1050)),input1,metric='cosine')
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

closest_movies_df.to_csv('closest_dist_meta_redo2.csv',index = False)

