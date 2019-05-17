#!/usr/bin/env python -W ignore::DeprecationWarning
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras import layers
from keras import backend
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
import numpy as np
import pickle
import os
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import datetime
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
import create_dataframe

NUMBER_OF_DATASETS=3
FILEPATH_DICT = {'kaggle': 'data/detecting_insults_kaggler/train.csv','dataworld': 'data/offensive_language_dataworld/data/labeled_data_squashed.csv'}
#, 'crowdsourced': 'data/model_input_data/crowd_sourced_processed.csv'

from sklearn.linear_model import LogisticRegression

now = datetime.datetime.now()
print('[',str(now),']', 'Starting demo')

def read_accuracies():
	model_fscore = None
	try:
		accuracy_store_file = open("./res/accuracy_with_F.txt", "r")
		model_fscore = {}
		content = accuracy_store_file.readlines()
		for line in content:
			accuracy_items = line.split(':')
			model_fscore[accuracy_items[0]] = float(accuracy_items[1][:-1])
	except IOError as e:
		print("Error accessing file", e)
	except Exception as e:
		print(e)
	return model_fscore

def normalize_dataset():
	model_fscore = read_accuracies()
	dataset_weights = {}
	for sources in FILEPATH_DICT.keys():
		dataset_weights[sources] = []
	for sources in FILEPATH_DICT.keys():
		denom_1 = denom_2 = denom_3 = 0
		for denom_sources in FILEPATH_DICT.keys():
			denom_1 = denom_1 + model_fscore["cnn_bow_" + denom_sources + "_fscore"]
			denom_2 = denom_2 + model_fscore["cnn_we_" + denom_sources + "_fscore"]
			denom_3 = denom_3 + model_fscore["cnn_we_pooling_" + denom_sources + "_fscore"]
		dataset_weights[sources].append(model_fscore["cnn_bow_" + sources + "_fscore"]/denom_1)
		dataset_weights[sources].append(model_fscore["cnn_we_" + sources + "_fscore"]/denom_2)
		dataset_weights[sources].append(model_fscore["cnn_we_pooling_" + sources + "_fscore"]/denom_3)
	print("Dataset weights based on accuracy")
	print(dataset_weights)
	return dataset_weights
		# dataset_weights[sources].append(model_fscore['cnn_bow_kaggle_fscore'])

def classify_tweet(df, list_of_tweets, dataset_weights, normalization=1):
	offensive_score = [0]*len(list_of_tweets)
	try:
		for source in df['source'].unique():
		    df_source = df[df['source'] == source]
		    sentences = df_source['tweet'].values
		    y = df_source['label'].values
		    now = datetime.datetime.now()
		    print('[',str(now),']', 'Processing started for source', source)
		    print("----------------------------------------------------------------")
		    print("----------------------------------------------------------------")
		    print(source.upper(), "SOURCE")
		    print("----------------------------------------------------------------")
		    print("----------------------------------------------------------------")
		    #splitting dataset into training and validation data
		    sentences_train, sentences_test, y_train, y_test = train_test_split(
		        sentences, y, test_size=0.25, random_state=1000)
		    
		    '''
		    CNN with BOW
		    '''
		    
		    vectorizer = CountVectorizer()
		    vectorizer.fit(sentences_train)
		    input_query = np.asarray(list_of_tweets)
		    input_query = vectorizer.transform(input_query)
		    
		    now = datetime.datetime.now()
		    print('[',str(now),']', 'Predicting with CNN BOW model prepared for source', source)
		    filename = './model/cnn_bow_' + source + '.h5'
		    loaded_model = load_model(filename)
		    predicted_value = loaded_model.predict(input_query)
		    cnt = 0
		    for i in predicted_value:
		    	float_equiv = i.astype(float)
		    	float_equiv = float_equiv[0]
		    	if float_equiv < 0.25:
		    		offensive_score[cnt] = offensive_score[cnt] - (1 - (1 - dataset_weights[source][0])*normalization)
		    	if float_equiv > 0.75:
		    		offensive_score[cnt] = offensive_score[cnt] + (1 - (1 - dataset_weights[source][0])*normalization)
		    	print(source, offensive_score[cnt])
		    	cnt = cnt+1
		    print(loaded_model.predict(input_query))
		    backend.clear_session()
		    
		    '''
		    #CNN with word embedding
		    '''

		    tokenizer = Tokenizer(num_words=5000)
		    tokenizer.fit_on_texts(sentences_train)
		    input_query = np.asarray(list_of_tweets)
		    input_query = tokenizer.texts_to_sequences(input_query)

		    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
		    maxlen = 100

		    input_query = pad_sequences(input_query, padding='post', maxlen=maxlen)
		    
		    now = datetime.datetime.now()
		    print('[',str(now),']', 'Predicting with CNN word embedded model prepared for source', source)

		    filename = './model/cnn_we_' + source + '.h5'
		    loaded_model = load_model(filename)
		    predicted_value = loaded_model.predict(input_query)
		    cnt = 0
		    for i in predicted_value:
		    	float_equiv = i.astype(float)
		    	float_equiv = float_equiv[0]
		    	if float_equiv < 0.25:
		    		offensive_score[cnt] = offensive_score[cnt] - (1 - (1 - dataset_weights[source][1])*normalization)
		    	if float_equiv > 0.75:
		    		offensive_score[cnt] = offensive_score[cnt] + (1 - (1 - dataset_weights[source][1])*normalization)
		    	cnt = cnt+1
		    print(loaded_model.predict(input_query))
		    backend.clear_session()
		    
		    #with GlobalMaxPooling1D layer to reduce number of features
		   
		    now = datetime.datetime.now()
		    print('[',str(now),']', 'Predicting with CNN word embedded model with global pooling prepared for source', source)
		    filename = './model/cnn_we_pooling_' + source + '.h5'
		    loaded_model = load_model(filename)
		    predicted_value = loaded_model.predict(input_query)
		    cnt = 0
		    for i in predicted_value:
		    	float_equiv = i.astype(float)
		    	float_equiv = float_equiv[0]
		    	if float_equiv < 0.25:
		    		offensive_score[cnt] = offensive_score[cnt] - (1 - (1 - dataset_weights[source][2])*normalization)
		    	if float_equiv > 0.75:
		    		offensive_score[cnt] = offensive_score[cnt] + (1 - (1 - dataset_weights[source][2])*normalization)
		    	print(offensive_score[cnt])
		    	cnt = cnt+1
		    print(loaded_model.predict(input_query))
		    backend.clear_session()
		for i in range(len(offensive_score)):
			if normalization == 0:
				offensive_score[i] = offensive_score[i]/(3*NUMBER_OF_DATASETS)
			else:
				offensive_score[i] = offensive_score[i]/3
		print(offensive_score)
	except FileNotFoundError as e:
		print("Error accessing file", e)
	except Exception as e:
		print(e)
	return( float(offensive_score[0]))
# input_query_list = ['shut up bitch', 'i love my mom', 'hi honey', 'you are a hoe', 'damn mama smack that']
# # input_query_list = ['you are a stupid moron you disgusting hag']

# dataset_weights = normalize_dataset()
# df = create_dataframe.setup_dataframe()
# classify_tweet(df, input_query_list, dataset_weights, normalization=1)

