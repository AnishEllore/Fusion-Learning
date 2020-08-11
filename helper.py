def generate_plots():
	
	precision = (tp+1)/float(tp+fp+1)
	recall = (tp+1)/float(tp+fn+1)
	f1 = 2*((precision*recall)/(precision+recall))
	print("Precision:{}, Recall:{}, F1-score:{} (f1 wrote to file)".format(precision, recall, f1))
