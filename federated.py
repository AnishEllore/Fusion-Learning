import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from distributions import Distribution
import tensorflow as tf
import plotly
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

class FederatedDistributions:
	def __init__(self,INPUT_SHAPE=20,test=True):
		# self.x = np.array(X)
		# self.y = np.array(Y)
		# self.X = X
		# self.Y = Y
		self.epoch_divider = 1
		self.INPUT_SHAPE = INPUT_SHAPE
		self.test = test
		if not os.path.isdir('results'):
			os.makedirs('results')
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
		plt.legend(['train'], loc='lower right')
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
		plt.legend(['train'], loc='lower right')
		if file_name!= None:
			name = 'results/'+file_name+' '+'lossVSepochs'
			plt.savefig(name+'.eps')
			plt.savefig(name+'.png')
		#plt.show()
		plt.clf()
	def train_model(self,x,y,Epochs=100):
		# Normal model building
		model_main = self.model_build()
		history = model_main.fit(x, y, epochs = int(Epochs/self.epoch_divider))
		print("************training evaluation**********")
		print(model_main.evaluate(x, y))
		print('***********done**************')
		return model_main, history
	
	def train_model_validation(self,x,y,x_val,y_val,Epochs=100):
		# Normal model building
		model_main = self.model_build()
		history = model_main.fit(x, y,validation_data=(x_val,y_val), epochs = int(Epochs/self.epoch_divider))
		print("************training evaluation**********")
		print(model_main.evaluate(x, y))
		print('***********done**************')
		return model_main, history

	def get_feature_distributions(self,data, features):
		distributions = []
		new_data = {}
		df_feature_fit = pd.DataFrame(columns=['feature','distribution','Pvalue'])
		for feature in features:
			dst = Distribution(test = self.test)
			# print(feature)
			# print(data[feature])
			dst.Fit(data[feature])
			generated_data, distribution_name, Pvalue = dst.Plot(data[feature])
			df_feature_fit = df_feature_fit.append({'feature':feature,'distribution':distribution_name,'Pvalue':1},ignore_index=True)
			distributions.append(distribution_name)
			new_data[feature] = generated_data
		# print(distributions)
		return distributions, new_data,df_feature_fit


	def data_distributions(self, X):
		feature_distributions, ndata,df_feature_fit = self.get_feature_distributions(X,X.columns)
		print(df_feature_fit)
		fig = genSankey(df_feature_fit,cat_cols=['feature','distribution'],value_cols='Pvalue',title='Feature Distributions')
		plotly.offline.plot(fig, validate=False)
	
	def build_setup(self,x,x_test,y,y_test):
		# x, x_test, y, y_test = train_test_split(self.x, self.y, test_size = 0.2, random_state = 0)
		original_model,history = self.train_model(x,y,200)
		self.show_plot(history,file_name='original_model',show=True)




		SPLIT_SIZE = 10
		X_train, Y_train = list(), list()
		X_new, Y_new = list(), list()
		local_model_node_testing_accuracies = []
		node_distributions = {}
		node_test_set=[]

		for i in range(0,SPLIT_SIZE):
			X_train.append(x[int((i*len(x)/SPLIT_SIZE)):(int((i+1)*len(x)/SPLIT_SIZE))])
			Y_train.append(y[(int(i*len(x)/SPLIT_SIZE)):(int((i+1)*len(x)/SPLIT_SIZE))])
			print('Running data on node ',i+1)
			x_curr, x_test_curr, y_curr, y_test_curr = train_test_split(X_train[i], Y_train[i], test_size = 0.2, random_state = 0)
			node_test_set.append((x_test_curr,y_test_curr))
			current_model, _ = self.train_model(x_curr,y_curr,200)
			#current_model = train_model(np.array(X_train[i]),np.array(Y_train[i]),1000)
			testing_accuracy = current_model.evaluate(x_test_curr,y_test_curr)
			local_model_node_testing_accuracies.append(testing_accuracy[1]*100) 
			print(testing_accuracy)
			# if int(testing_accuracy[1]*100) <91:
			# 	print('##############BINGO################',i)
			# 	testing_accuracies_temp.append(testing_accuracy[1])
			# 	continue
			x_df = pd.DataFrame(X_train[i])
			feature_distributions, new_data,_ = self.get_feature_distributions(x_df, x_df.columns)
			node_distributions[i] = feature_distributions
			
			
			X_new.extend( pd.DataFrame(new_data).values)
			Y_new.extend(current_model.predict_classes(pd.DataFrame(new_data).values))



		new_data_X = np.array(X_new)
		new_data_Y = np.array(Y_new)

		x_new, x_new_test, y_new, y_new_test = train_test_split(new_data_X,new_data_Y, test_size = 0.2, random_state = 0)

		scaler = MinMaxScaler()
		new_data_X = scaler.fit_transform(new_data_X)
		
		model_new_main, global_history = self.train_model_validation(new_data_X,new_data_Y,scaler.transform(x),y,200)
		self.show_plot(global_history,file_name='global_model',show=True)
		global_model_node_testing_accuracies = []
		for node_set in node_test_set:
			node_x_test, node_y_test = node_set
			testing_accuracy = model_new_main.evaluate(node_x_test, node_y_test)
			global_model_node_testing_accuracies.append(testing_accuracy[1]*100)



		results = {}

		results['new model testing accuracy'] = model_new_main.evaluate(scaler.transform(x_new_test),y_new_test)[1]*100

		results['with new model and old data points'] = model_new_main.evaluate(scaler.transform(x_test), y_test)[1]*100

		results['With old model and old All'] = original_model.evaluate(x_test, y_test)[1]*100

		with open('results/out.txt', 'w') as f:
			# feature_distributions, ndata,df_feature_fit = self.get_feature_distributions(self.X,self.X.columns)
			# print(df_feature_fit)
			# fig = genSankey(df_feature_fit,cat_cols=['feature','distribution'],value_cols='Pvalue',title='Feature Distributions')
			# plotly.offline.plot(fig, validate=False)
			# print('Data feature distributions')
			# print(feature_distributions)
			print('local model testing accuracies over nodes',file=f)
			print(local_model_node_testing_accuracies,file=f)
			# print('Distributions at each node')
			# print(node_distributions)
			print('global model testing accuracy over nodes',file=f)
			print(global_model_node_testing_accuracies,file=f)


			print('printing other results',file=f)
			print(results,file=f)
		comparision_save = [list(i) for i in zip(local_model_node_testing_accuracies,global_model_node_testing_accuracies)]
		comparision_save = pd.DataFrame(comparision_save)
		comparision_save.to_csv('results/local_vs_global',index=False)
		return history.history['accuracy'],global_history.history['val_accuracy']