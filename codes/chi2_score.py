import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from matplotlib import pyplot


def prepare_inputs(d):
	oe = OrdinalEncoder()
	oe.fit(d)
	d = oe.transform(d)
	return d

data = pd.read_csv("data0.csv")
data['Delay'] = data['Delay'].astype(int)

#categorize delay
data.loc[data['Delay']<=15,'Delay'] = 1
data.loc[data['Delay']>15,'Delay'] = 0

# print(data.loc[data['Delay']==0])

dataset = data.values
dataset = prepare_inputs(dataset)
columns = data.columns
columns = columns[:-1]
y = dataset[:,-1]
x = dataset[:,:-1]

# selecting k Best
fs = SelectKBest(score_func=chi2, k=5)
fs.fit(x,y)


score = fs.scores_
df = pd.Series(data = score,index=columns)
print(df.sort_values(ascending = True))

#bar plot based on chi_square correlation 
df.plot(kind='bar',color='blue', fontsize = 18)
plt.suptitle('CHI_2 Rank of features with Flight Status', fontsize=25)
plt.savefig('chi_2.png',bbox_inches='tight')
plt.show()
