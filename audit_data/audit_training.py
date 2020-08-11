import sys
sys.path.append("..")
from federated import FederatedDistributions
from federated_google import FederatedLearning
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

scaler = StandardScaler()

df = pd.read_csv('audit_risk.csv')
# df2 = pd.read_csv('trial.csv')
df = df.apply (pd.to_numeric, errors='coerce')
df = df.dropna()

df = shuffle(df).reset_index(drop=True)

X = df.drop(labels=['Risk'],axis=1).astype(np.float32)
'''
# if preprocessing is needed
c = scaler.fit_transform(X)
X = pd.DataFrame(c)
'''
Y = df['Risk']


c = scaler.fit_transform(X)
X = pd.DataFrame(c)

x, x_test, y, y_test = train_test_split(np.array(X), np.array(Y), test_size = 0.2, random_state = 0)

FD = FederatedDistributions(INPUT_SHAPE=len(X.columns),test=False)
main_model_history,global_model_history = FD.build_setup(x, x_test, y, y_test)
# FD.data_distributions()

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