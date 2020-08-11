import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from distributions import Distribution
import tensorflow as tf
import plotly
import os

def genSankey(df,cat_cols=[],value_cols='',title='Sankey Diagram'):
	# maximum of 6 value cols -> 6 colors
	colorPalette = ['#4B8BBE','#306998','#FFE873','#FFD43B','#646464']
	labelList = []
	colorNumList = []
	for catCol in cat_cols:
		labelListTemp =  list(set(df[catCol].values))
		colorNumList.append(len(labelListTemp))
		labelList = labelList + labelListTemp
		
	# remove duplicates from labelList
	labelList = list(dict.fromkeys(labelList))
	
	# define colors based on number of levels
	colorList = []
	for idx, colorNum in enumerate(colorNumList):
		colorList = colorList + [colorPalette[idx]]*colorNum
		
	# transform df into a source-target pair
	for i in range(len(cat_cols)-1):
		if i==0:
			sourceTargetDf = df[[cat_cols[i],cat_cols[i+1],value_cols]]
			sourceTargetDf.columns = ['source','target','count']
		else:
			tempDf = df[[cat_cols[i],cat_cols[i+1],value_cols]]
			tempDf.columns = ['source','target','count']
			sourceTargetDf = pd.concat([sourceTargetDf,tempDf])
		sourceTargetDf = sourceTargetDf.groupby(['source','target']).agg({'count':'sum'}).reset_index()
		
	# add index for source-target pair
	sourceTargetDf['sourceID'] = sourceTargetDf['source'].apply(lambda x: labelList.index(x))
	sourceTargetDf['targetID'] = sourceTargetDf['target'].apply(lambda x: labelList.index(x))
	
	# creating the sankey diagram
	data = dict(
		type='sankey',
		node = dict(
		  pad = 15,
		  thickness = 20,
		  line = dict(
			color = "black",
			width = 0.5
		  ),
		  label = labelList,
		  color = colorList
		),
		link = dict(
		  source = sourceTargetDf['sourceID'],
		  target = sourceTargetDf['targetID'],
		  value = sourceTargetDf['count']
		)
	  )
	
	layout =  dict(
		title = title,
		font = dict(
		  size = 10
		)
	)
	   
	fig = dict(data=[data], layout=layout)
	return fig

class FederatedLearning:
	def __init__(self,INPUT_SHAPE=20,test=True):
		# self.x = np.array(X)
		# self.y = np.array(Y)
		# self.X = X
		# self.Y = Y
		self.epoch_divider = 1
		self.INPUT_SHAPE = INPUT_SHAPE
		self.test = test
		if not os.path.isdir('federated_google_results'):
			os.makedirs('federated_google_results')
		if test == True:
			self.epoch_divider = 100
		# self.build_setup()
	def model_build(self) :
		model = tf.keras.models.Sequential()
		model.add(tf.keras.layers.Dense(10, input_shape = (self.INPUT_SHAPE,), activation='relu', use_bias = False))
		model.add(tf.keras.layers.Dense(10, activation='relu', use_bias = False))
		model.add(tf.keras.layers.Dense(2, activation='softmax'))
		model.compile(optimizer='adam',
					  loss='sparse_categorical_crossentropy',
					  metrics=['accuracy'])
		
		return model

	def show_plot(self,history,show=False,file_name=None):
		data1 = history.history['accuracy']
		data2 = history.history['loss']
		x_axis = [i for i in range(1,len(data1)+1)]
		plt.figure()
		plt.plot(x_axis,data1)
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train'], loc='upper left')
		if file_name!= None:
			name = 'results/'+file_name+' '+'accuracyVSepochs'
			plt.savefig(name+'.eps')
			plt.savefig(name+'.png')
		#plt.show()
		plt.clf()
		# summarize history for loss
		plt.figure()
		plt.plot(x_axis,data2)
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train'], loc='upper left')
		if file_name!= None:
			name = 'results/'+file_name+' '+'lossVSepochs'
			plt.savefig(name+'.eps')
			plt.savefig(name+'.png')
		#plt.show()
		plt.clf()


	def build_setup(self,x, x_test, y, y_test,Epochs = 200):
		#x, x_test, y, y_test = train_test_split(self.x, self.y, test_size = 0.2, random_state = 0)

		if self.test:
			Epochs=10
		SPLIT_SIZE = 10
		X_train, Y_train = list(), list()
		
		node_test_set=[]
		node_data_set =[]
		for i in range(0,SPLIT_SIZE):
			X_train.append(x[int((i*len(x)/SPLIT_SIZE)):(int((i+1)*len(x)/SPLIT_SIZE))])
			Y_train.append(y[(int(i*len(x)/SPLIT_SIZE)):(int((i+1)*len(x)/SPLIT_SIZE))])
			x_curr, x_test_curr, y_curr, y_test_curr = train_test_split(X_train[i], Y_train[i], test_size = 0.2, random_state = 0)
			node_test_set.append((x_test_curr,y_test_curr))
			node_data_set.append((x_curr,y_curr))
		training_accuracies = []
		global_model = self.model_build()

		for i in range(0,Epochs):
			models=[]
			for i in range(0,SPLIT_SIZE):
				print('Running data on node ',i+1)
				x_curr, y_curr = node_data_set[i]
				current_model = global_model
				current_model.fit(x_curr, y_curr, epochs = 1)
				models.append(current_model)
			weights = [model.get_weights() for model in models]
			new_weights = list()

			for weights_list_tuple in zip(*weights):
				new_weights.append(
					np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)]))

			global_model.set_weights(new_weights)
			training_accuracy = global_model.evaluate(x,y)[1]
			training_accuracies.append(training_accuracy)

		plt.figure()
		plt.plot(training_accuracies)
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train'], loc='upper left')
		plt.savefig('federated_google_results/federatedLearning_accuracyVsepochs.png')
		plt.savefig('federated_google_results/federatedLearning_accuracyVsepochs.eps')
		plt.clf()
		results = {}

		results['global model testing accuracy'] = global_model.evaluate(x_test,y_test)[1]*100


		with open('federated_google_results/out.txt', 'w') as f:
			print('printing other results',file=f)
			print(results,file=f)
		
		comparision_save = pd.DataFrame(training_accuracies)
		comparision_save.to_csv('federated_google_results/training_accuracies',index=False)
		return training_accuracies
