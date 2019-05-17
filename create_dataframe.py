import pandas as pd

FILEPATH_DICT = {'kaggle':   'data/detecting_insults_kaggler/train.csv','dataworld': 'data/offensive_language_dataworld/data/labeled_data_squashed.csv'}


def setup_dataframe():
	df_list = []
	source = "kaggle"
	filepath = FILEPATH_DICT["kaggle"]
	#df = pd.read_csv(filepath, names=['rev_id', 'comment year','logged_in',   'ns',  'sample',  'split'], sep='\t')
	df = pd.read_csv(filepath, names=['label', 'date','tweet'], sep=',',header=0)
	df['source'] = source  # Add another column filled with the source name
	df_list.append(df)
	df = pd.concat(df_list)
	df = df.drop(['date'], axis=1)
	source = "dataworld"
	filepath = FILEPATH_DICT["dataworld"]
	#df = pd.read_csv(filepath, names=['rev_id', 'comment year','logged_in',   'ns',  'sample',  'split'], sep='\t')
	df = pd.read_csv(filepath, names=['id', 'count','hate_speech', 'offensive_language','neither','class', 'tweet', 'label'], sep=',',header=0)
	df['source'] = source  # Add another column filled with the source name
	df_list.append(df)
	df = pd.concat(df_list)
	df = df.drop(['count','hate_speech', 'offensive_language','neither'], axis=1)
	return df