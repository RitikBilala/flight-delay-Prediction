# evaluation of a model fit using chi squared input features
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from numpy import cov

def corr(x,y):
    dfy = pd.DataFrame(y)
    dfx = pd.DataFrame(x)
    frame = [dfx,dfy]
    d=pd.concat(frame,axis=1)
    corr= d.corr().values
    corr= corr[:,-1]
    corr = corr[:-1]
    print(corr)
    return corr

def select_features(X_, y_,k_):
	fs = SelectKBest(score_func=corr, k= k_)
	fs.fit(X_, y_)
	X_fs = fs.transform(X_)
	return X_fs ,fs

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
df4 = pd.get_dummies(data['DISTANCE'],prefix='d_')
df5 = pd.get_dummies(data['FL_NUM'])
df6 = pd.get_dummies(data['ORIGIN'])
df7 =pd.DataFrame(data=data['Weather'].values,columns=['Weather'])
df8 = pd.get_dummies(data['DAY_WEEK'],prefix='day')
df9 = pd.get_dummies(data['DEST'])
df = df.astype(int)

frames1 = [df1,df2,df3,df4,df5,df6,df7,df8,df9]

d=pd.concat(frames1,axis=1)
print(d)
X =d.values
y= df['Flight Delay']

k=5
X_fs,fs= select_features(X, y,k)

X_train, X_test, y_train, y_test = train_test_split(X_fs, y, test_size=0.4, random_state= 36)

model = LogisticRegression(solver='lbfgs',max_iter=10e8,penalty = 'l2',fit_intercept=True)
model.fit(X_train, y_train)
# evaluate the modelR
yhat = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, yhat)
print('Accuracy: %.2f' % (accuracy*100))
print(model.coef_)

score = fs.scores_
score = pd.Series(data = score,index=d.columns)
score = score.sort_values(ascending = False)
bestk = score[0:5]
print(score[0:k])


bestk.plot(kind='bar',color=['r','b','g','y','fuchsia','maroon'],fontsize = 15)
plt.title('Best_5 features based on correlation to delay', fontsize=25)
plt.xticks(rotation=0)
plt.savefig('Best_5_correlations.png', dpi=500,bbox_inches='tight')
plt.show()
