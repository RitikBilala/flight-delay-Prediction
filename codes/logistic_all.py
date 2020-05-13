# evaluation of a model fit using chi squared input features
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


	# load the dataset as a pandas DataFrame
data = pd.read_csv('data0.csv')
data.loc[data['Delay']<=15,'Delay'] = 0
data.loc[data['Delay']>15,'Delay'] = 1
data.loc[data['Flight Status']=='ontime','Flight Status'] = 0
data.loc[data['Flight Status']=='delayed','Flight Status'] = 1
data['DEPARTURE'] = data['DEPARTURE'].astype(str)
data['DISTANCE'] = data['DISTANCE'].astype(str)
data['DAY_WEEK'] = data['DAY_WEEK'].astype(str)

df =pd.DataFrame(data=data['Flight Status'].values,columns=['Flight Delay'])
df1 =pd.DataFrame(data=data['Delay'].values,columns=['Dep. Delay'])
df2 = pd.get_dummies(data['DEPARTURE'],prefix='slot')
df3 = pd.get_dummies(data['CARRIER'])
df4 = pd.get_dummies(data['DISTANCE'])
df5 = pd.get_dummies(data['FL_NUM'],prefix='FL_')
df6 = pd.get_dummies(data['ORIGIN'])
df7 =pd.DataFrame(data=data['Weather'].values,columns=['Weather'])
df8 = pd.get_dummies(data['DAY_WEEK'],prefix='day')
df9 = pd.get_dummies(data['DEST'])
df = df.astype(int)

frames1 = [df1,df2,df3,df4,df5,df6,df7,df8,df9]

d=pd.concat(frames1,axis=1)
X =d.values
y= df['Flight Delay']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state= 36)

model = LogisticRegression(solver='lbfgs',max_iter=10e8,penalty = 'l2',fit_intercept=True)
model.fit(X_train, y_train)
# evaluate the modelR
yhat = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, yhat)
print('Accuracy: %.2f' % (accuracy*100))
print(model.coef_)

arr = np.array(model.coef_.reshape(model.coef_.size))
index = d.columns

x_coef = pd.Series(arr, index = d.columns)
mask=abs(x_coef)>1
selected_x = x_coef[mask]

selected_x.plot(kind='bar',color=(selected_x > 0).map({True: 'r',False: 'g'}),fontsize = 15)
plt.title('Features with abs. LogisticRegression coeff. > 1 ', fontsize=25)
plt.savefig('LogisticRegressioncoef .png', dpi=500, bbox_inches='tight')
plt.show()
