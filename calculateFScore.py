
def calculate_F_Score():
	accuracy_store_file_read = open("./res/accuracy.txt", "r")
	accuracy_store_file_write = open("./res/accuracy_with_F.txt", "w+")
	model_accuracy = {}
	content = accuracy_store_file_read.readlines()
	for line in content:
		accuracy_items = line.split(':')
		model_accuracy[accuracy_items[0]] = float(accuracy_items[1][:-1])
	print(model_accuracy)
	types = ["cnn_bow_kaggle", "cnn_we_kaggle", "cnn_we_pooling_kaggle", "cnn_bow_dataworld", "cnn_we_dataworld", "cnn_we_pooling_dataworld"]
	for i in types:
		model_accuracy[i+"_fscore"] = 2*float(model_accuracy[i+"_precision"])*float(model_accuracy[i+"_recall"]) / float((model_accuracy[i+"_precision"])+float(model_accuracy[i+"_recall"]))
	print(model_accuracy)
	accuracy_store_file_write.write(str(model_accuracy))

calculate_F_Score()