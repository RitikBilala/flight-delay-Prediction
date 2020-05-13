# evaluation of a model fit using chi squared input features
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
df5 = pd.get_dummies(data['FL_NUM'])
df6 = pd.get_dummies(data['ORIGIN'])
df7 =pd.DataFrame(data=data['Weather'].values,columns=['Bad Weather'])
df8 = pd.get_dummies(data['DAY_WEEK'])
df9 = pd.get_dummies(data['DEST'])
df = df.astype(int)
frames1 = [df1,df]
frames2 = [df2,df]
frames3 = [df3,df]
frames4 = [df4,df]
frames5 = [df5,df]
frames6 = [df6,df]
frames7 = [df7,df]
frames8 = [df8,df]
frames9 = [df9,df]
d1=pd.concat(frames1,axis=1)
d2=pd.concat(frames2,axis=1)
d3=pd.concat(frames3,axis=1)
d4=pd.concat(frames4,axis=1)
d5=pd.concat(frames5,axis=1)
d6=pd.concat(frames6,axis=1)
d7=pd.concat(frames7,axis=1)
d8=pd.concat(frames8,axis=1)
d9=pd.concat(frames9,axis=1)
print(d1)
print(d2)
print(d3)
print(d4)
print(d5)
print(d6)
print(d7)
print(d8)
print(d9)

c1 = d1.corr()['Flight Delay'].drop(['Flight Delay'])
c2 = d2.corr()['Flight Delay'].drop(['Flight Delay'])
c3 = d3.corr()['Flight Delay'].drop(['Flight Delay'])
c4 = d4.corr()['Flight Delay'].drop(['Flight Delay'])
c5 = d5.corr()['Flight Delay'].drop(['Flight Delay'])
c6 = d6.corr()['Flight Delay'].drop(['Flight Delay'])
c7 = d7.corr()['Flight Delay'].drop(['Flight Delay'])
c8 = d8.corr()['Flight Delay'].drop(['Flight Delay'])
c9 = d9.corr()['Flight Delay'].drop(['Flight Delay'])

print(c1)
print(c2)
print(c3)
print(c4)
print(c5)
print(c6)
print(c7)
print(c8)
print(c9)
 
c1.plot(kind='bar',color=(c1 > 0).map({True: 'b',False: 'r'}),fontsize = 15)
plt.title('Departure delay correlation', fontsize=25)
plt.xticks(rotation=0)
plt.savefig('Correlation Graphs//Departure delay.png', dpi=500, bbox_inches='tight')
plt.show()
plt.clf()

c2.plot(kind='bar',color=(c2 > 0).map({True: 'r',False: 'g'}),fontsize = 15)
plt.title('Departure time slot correlation', fontsize=25)
plt.savefig('Correlation Graphs//Departure time.png', dpi=500,bbox_inches='tight')
plt.show()
plt.clf()


c3.plot(kind='bar',color=(c3 > 0).map({True: 'r',False: 'g'}),fontsize = 15)
plt.title('Carrier correlation', fontsize=25)
plt.xticks(rotation=0)
plt.savefig('Correlation Graphs//carrier.png',dpi=500, bbox_inches='tight')
plt.show()
plt.clf()


c4.plot(kind='bar',color=(c4 > 0).map({True: 'r',False: 'g'}),fontsize = 15)
plt.title('Distance Correlation', fontsize=25)
plt.xticks(rotation=0)
plt.savefig('Correlation Graphs//distance.png', dpi=500,bbox_inches='tight')
plt.show()
plt.clf()

c5.plot(kind='bar',color=(c5 > 0).map({True: 'r',False: 'g'}))
plt.title('Flight Number Correlation', fontsize=25)
plt.savefig('Correlation Graphs//Flight Number.png', dpi=500,bbox_inches='tight')
plt.show()
plt.clf()

c6.plot(kind='bar',color=(c6 > 0).map({True: 'r',False: 'g'}),fontsize = 15)
plt.title('Origin based Delay', fontsize=25)
plt.xticks(rotation=0)
plt.savefig('Correlation Graphs//origin.png', dpi=500,bbox_inches='tight')
plt.show()
plt.clf()

c7.plot(kind='bar',color=(c7 > 0).map({True: 'b',False: 'g'}),fontsize = 15)
plt.title('Weather correlation', fontsize=25)
plt.xticks(rotation=0)
plt.savefig('Correlation Graphs//Weather', dpi=500,bbox_inches='tight')
plt.show()
plt.clf()

c8.plot(kind='bar',color=(c8 > 0).map({True: 'r',False: 'g'}),fontsize = 15)
plt.title('Day of Week correlation', fontsize=25)
plt.xticks(rotation=0)
plt.savefig('Correlation Graphs//DAY_WEEK', dpi=500,bbox_inches='tight')
plt.show()
plt.clf()

c9.plot(kind='bar',color=(c9 > 0).map({True: 'r',False: 'g'}),fontsize = 15)
plt.title('Destination correlation', fontsize=25)
plt.xticks(rotation=0)
plt.savefig('Correlation Graphs//Destination .png', dpi=500, bbox_inches='tight')
plt.show()
plt.clf()