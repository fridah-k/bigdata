import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D # 3d plot
from termcolor import colored as cl # text customization

from sklearn.preprocessing import StandardScaler # data normalization
from sklearn.cluster import KMeans # K-means algorithm
plt.rcParams['figure.figsize'] = (20, 10)
sns.set_style('whitegrid')

df= pd.read_csv("/home/fridah/Downloads/customer_segmentation.csv")
df.drop('Unnamed: 0', axis= 1, inplace= True)
df.set_index('Customer Id', inplace = True)
print(df.head(10))
print(df.describe())
print(df.isnull().count())
# df.NaN.fill(value=0,subset=['Defaulted'])
# print(df.head(10))

# sns.distplot(df['Age'], color= 'orange')
# plt.title('AGE DISTRIBUTION', fontsize= 18)
# plt.xlabel('Age', fontsize= 16)
# plt.ylabel('Frequency', fontsize= 16)
# plt.xticks(fontsize= 14)
# plt.yticks(fontsize = 14)
# plt.savefig('age_distribution.png')
# plt.show()
# the ouput:
# the age with the highest number of customers is 30 to 50 and the least is btw 20 to 30
# sns.distplot(df['Age'], color= 'blue')
# plt.title('INCOME DISTRIBUTION', fontsize= 18)
# plt.xlabel('Age', fontsize= 16)
# plt.ylabel('Income', fontsize= 16)
# plt.xticks(fontsize= 14)
# plt.yticks(fontsize = 14)
# plt.savefig('income_distribution.png')
# plt.show()
# the output is that the age with the highest income is between 30-50 and the least is btw 20-30

#CREDIT Card Default cases
# default= df[df['Defaulted'] == 1.0]
# non_default= df[df['Defaulted'] == 0.0]

# print(cl('..........', attrs = ['bold']))
# print(cl('Number of Default cases are {}'.format(len(default)), attrs= ['bold']))
# print(cl('..........', attrs = ['bold']))
# print(cl('Number of Non_Default cases are {}'.format(len(non_default)), attrs= ['bold']))
# print(cl('..........', attrs = ['bold']))
# print(cl('Percentage of Default cases is {:.0%}'.format(len(default)/len(non_default)), attrs= ['bold']))
# print(cl('..........', attrs = ['bold']))

# sns.countplot(df['Defaulted'], palette= ['coral', 'deepskyblue'], edgecolor= 'darkgrey')
# plt.title('Credit card default cases(1) and non-default cases(0)', fontsize= 18)
# plt.xlabel('Default value', fontsize= 16)
# plt.ylabel('Number of People', fontsize= 16)
# plt.xticks(fontsize= 14)
# plt.yticks(fontsize= 14)

# plt.savefig('default_cases.png')
# plt.show()

# Years employed vs Income
# area= df.DebtIncomeRatio 
# sns.scatterplot('Years Employed', 'Income', 
#                  data= df, 
#                  s= 50, 
#                  alpha= 0.6, 
#                  edgecolor= 'white', 
#                  hue= 'Defaulted',
#                  palette= 'spring')
# # s is the area of each bubble
# plt.title('YEARS EMPLOYED/ INCOME', fontsize= 18)
# plt.xlabel('Years Employed', fontsize= 16)
# plt.ylabel('Income', fontsize= 16)
# plt.xticks(fontsize = 14)
# plt.yticks(fontsize = 14)
# plt.legend(loc = 'upper left', fontsize = 14)

# plt.savefig('y_income.png')
# plt.show()
# Normalization of the Data
x= df.values
x= np.nan_to_num(x)
sc= StandardScaler()

cluster_data= sc.fit_transform(x)
print(cl('Cluster data samples:', attrs= ['bold']), cluster_data[:5])

#MODELLING USING KMEANS ALGORITHMN
# in the modelling we get the labels for training in the model
clusters = 3
model= KMeans(init= 'k-means++',n_clusters= clusters, n_init= 12)
model.fit(x)
labels= model.labels_
print(cl(labels[:100], attrs= ['bold']))
# getting the cluster value of each row
df['cluster_num']= labels
print(df.head())
print(df.groupby('cluster_num').mean())
# let visualize the age and income using cluster and bubble plot
sns.scatterplot('Age', 'Income', data=df, s= 100, hue= 'cluster_num', palette= 'spring', alpha= 0.6, edgecolor= 'darkgrey')
plt.title('AGE/INCOME(CLUSTERED)', fontsize= 18)
plt.xlabel('Age', fontsize= 16)
plt.ylabel('Income', fontsize= 16)
plt.xticks(fontsize= 14)
plt.yticks(fontsize= 14)
plt.legend(loc= 'upper left', fontsize= 14)
plt.savefig('cluster_age_income.png')
plt.show()
# the output is that each customer has been grouped according to cluster they belong

#We can produce a three-dimensional scatter plot using the ‘mplot3d’ package.
fig= plt.figure(1)
plt.clf()
ax= Axes3D(fig, rect= [0, 0, .95, 1], elev= 48, azim= 134)
plt.cla()
ax.scatter(df['Edu'], ['Age'], ['Income'],
             c= df['cluster_num'],
             s= 200,
             cmap= 'spring',
             alpha= 0.5,
             edgecolor= 'darkgrey')
ax.set_xlabel('Education', fontsize= 16)
ax.set_ylabel('Age', fontsize= 16)
ax.set_zlabel('Income', fontsize= 16)
plt.savefig('3D_plot.png')
plt.show()














