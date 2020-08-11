import sys
sys.path.append("..")
from federated import FederatedDistributions
from federated_google import FederatedLearning
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df = pd.read_csv("./breast-cancer-wisconsin.data",header=None)
df = shuffle(df).reset_index(drop=True)

scaler = StandardScaler()
# scaler = MinMaxScaler()

LABEL = df.columns[-1]
df=df[df[6]!='?']
df.dropna()
x = df.drop([df.columns[0], LABEL], axis=1)
print(x)
#x = scaler.fit_transform(x)
x = x.astype(np.float16)
X = x
Y = df[LABEL]
y = []
for i,j in enumerate(Y):
	if j == 2:
		y.append(0)
	else:
		y.append(1)
Y=y

X_Distributions = X
c = scaler.fit_transform(X)
X = pd.DataFrame(c)

x, x_test, y, y_test = train_test_split(np.array(X), np.array(Y), test_size = 0.2, random_state = 0)

FD = FederatedDistributions(INPUT_SHAPE=len(X.columns),test=False)
main_model_history,global_model_history = FD.build_setup(x, x_test, y, y_test)
# X_Distributions.columns = ['Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses']
# X.columns = X_Distributions.columns
# FD.data_distributions(X)

FL = FederatedLearning(INPUT_SHAPE=len(X.columns),test=False)
fl_model_history = FL.build_setup(x, x_test, y, y_test)

x_axis = [i for i in range(1,len(main_model_history)+1)]
plt.figure()
plt.plot(x_axis,main_model_history, label='Central Learning')
plt.plot(x_axis,global_model_history, label='Fusion Learning')
plt.plot(x_axis,fl_model_history,label='Federated Learning')
plt.title('models accuracy comparision')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='lower right')

name = 'results/'+'all_approaches_accuracy_vs_epochs'
plt.savefig(name+'.eps')
plt.savefig(name+'.png')
#plt.show()
plt.clf()