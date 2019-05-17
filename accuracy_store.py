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