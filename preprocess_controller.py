import preprocess

import csv
with open('./data/detecting_insults_kaggler/train.csv','r') as readFile:
	writeFile = open('./data/model_input_data/kaggle_processed.csv', 'w+')
	writeFile = csv.writer(writeFile)
	data = csv.reader(readFile)
	print("Cleaning data now. Might take a while...")
	for row in data:
		print(row[2])
		# print(row[2])
		new_text = preprocess.cleanup(row[2])
		print(new_text)
		row[2] = new_text
		print(row)
		writeFile.writerow(row)
print("Cleaned file kaggle/train.csv")
print("Cleaned file in kaggle/train_processed.csv")

with open('./data/offensive_language_dataworld/data/labeled_data_squashed.csv','r') as readFile:
	writeFile = open('./data/model_input_data/dataworld_processed.csv', 'w+')
	writeFile = csv.writer(writeFile)
	data = csv.reader(readFile)
	print("Cleaning data now. Might take a while...")
	for row in data:
		print(row[6])
		# print(row[2])
		new_text = preprocess.cleanup(row[6])
		print(new_text)
		row[6] = new_text
		print(row)
		writeFile.writerow(row)
print("Cleaned file dataworld/data/labeled_data_squashed.csv")
print("Cleaned file dataworld/data/labeled_data_squashed_processed.csv")


with open('./data/crowd_sourced_2.csv','r') as readFile:
	writeFile = open('./data/model_input_data/crowd_sourced_processed.csv', 'w+')
	writeFile = csv.writer(writeFile)
	data = csv.reader(readFile)
	print("Cleaning data now. Might take a while...")
	for row in data:
		print(row[0])
		# print(row[2])
		new_text = preprocess.cleanup(row[0])
		print(new_text)
		row[0] = new_text
		# print(row)
		writeFile.writerow(row)
print("Cleaned file dataworld/data/crowd_sourced.csv")
print("Cleaned file dataworld/data/crowd_sourced_processed.csv")