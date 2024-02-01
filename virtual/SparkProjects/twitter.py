import numpy as np
import pandas as pd
import re
import string
# plotting and visualizing
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
# import nltk
import nltk
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
# sklearn dependancies
from sklearn.svm import LinearSVC # svc i support vector classifier
from sklearn.naive_bayes import BernoulliNB #BernoulliNB is designed for binary/boolean features.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #common algorithm to transform
#text into a meaningful representation of numbers which is used to fit machine algorithm for prediction.
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
# load the data
DATASET_COLUMNS=['target','ids','date','flag','user','text']
DATASET_ENCODING = "ISO-8859-1"
# encoding is used converting binary into values which the machine understands.
#it is a character standard encoding
df= pd.read_csv('/home/fridah/Downloads/twitter.csv', encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
print(df.sample(5))
print(df.head())
## EXPLANATORY DATA ANALYSIS
print(df.columns)
# lenghth of the dataset
# print('length of data is', len(df))
# shape of the dataset
# print(df.shape)
# data information
# print(df.info())
# the data types
# print(df.dtypes)
# check for null values
# print(df.isnull().any(axis=1))
# The number of Rows and columns in the dataset
# print('Count of columns in the dataset is: ', len(df.columns))
# print('Count of rows in the dataset is: ', len(df))
# check unique target values
# print(df['target'].unique())
# check for non unique values
# print(df['target'].nunique())
## DATA VISUALIZATION
# ax= df.groupby('target').count().plot(kind='bar', title= 'Distribution of tweets',legend= 'False')
# ax.set_xticklabels(['Negative', 'Positive'], rotation=0)
# plt.savefig('tweet distribution')
# plt.show()
# text, sentiment = list(df['text']), list(df['target'])

# sns.countplot(x='target', data=df)
# plt.show()

# # Data preprocessing
# # this is where we get rid of unwanted items.Remove stopwords, emojis, noise that is puntuation marks,
# # repeating words, convertingtext to lowercase and performing stemming and lemmatization
#Selecting the text and Target column for our further analysis
data= df[['text', 'target']]
#Replacing the values to ease understanding. (Assigning 1 to Positive sentiment 4)
data['target']= data['target'].replace(4, 1)
# printing unique values of target variables
data['target'].unique()
# separating positive and negative tweets
data_pos= data[data['target'] == 1]
data_neg= data[data['target'] == 0]
# take a quarter of the data so that we can run the machine easily
data_pos= data_pos.iloc[:int(20000)]
data_neg= data_neg.iloc[:int(20000)]
# combining positive and negative tweets
dataset= pd.concat([data_pos, data_neg])
# changing the text to lower case
dataset['text']= dataset['text'].str.lower()
print(dataset['text'].head())
# Defining the stopwords list
stopwordlist= ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']
# clean and remove this stopwords from the tweets
STOPWORDS= set(stopwordlist) # the set()used to convert any of the iterable 
#to sequence of iterable elements with distinct elements. That means it goes through every word one by one just like looping
def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split()if word not in STOPWORDS])
    # in this line of return 
    # the code explains an empty string " " is being joined with another string to form one string
dataset['text']= dataset['text'].apply(lambda text: cleaning_stopwords(text))
print(dataset['text'].head())

# cleaning and removing puntuations
english_punctuations= string.punctuation
punctuations_list= english_punctuations
def cleaning_punctuations(text):
    translator= str.maketrans('', '', punctuations_list)
    return text.translate(translator)
dataset['text']= dataset['text'].apply(lambda x: cleaning_punctuations(x))
print(dataset['text'].head())

# cleaning and removing reppeating words
def cleaning_repeating_char(text):
    return re.sub(r'(.)1+', r'1', text)
dataset['text']= dataset['text'].apply(lambda x: cleaning_repeating_char(x))
print(dataset['text'].head())

# cleaning and removing urls
def cleaning_URLS(data):
    return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',data)
dataset['text']= dataset['text'].apply(lambda x: cleaning_URLS(x))
print(dataset['text'].head())

# removing numeric numbers
def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)
dataset['text']= dataset['text'].apply(lambda x: cleaning_numbers(x))
print(dataset['text'].head())

# tokenazation of the tweet text
tokenizer = RegexpTokenizer(r'w+')
dataset['text'] = dataset['text'].apply(tokenizer.tokenize)
print(dataset['text'].head())

# applying stemming on the data
st= nltk.PorterStemmer()
def stemming_on_text(data):
    text= [st.stem(word) for word in data]
    return data
dataset['text']= dataset['text'].apply(lambda x: stemming_on_text(x))
print(dataset['text'].head())

#applying lemmatizer
lm= nltk.WordNetLemmatizer()
def lemmatizer_on_text(data):
    text= [lm.lemmatize(word) for word in data]
    return data
dataset['text']= dataset['text'].apply(lambda x: lemmatizer_on_text(x))
print(dataset['text'].head())
#separating  input feature and label
X= data.text
y= data.target
# # plot cloud of words for negative tweets
data_neg= data['text']
wordcloud= WordCloud(max_words= 100, width= 1600, height= 800, collocations=False).generate(" ".join(data_neg))
plt.figure(figsize= (20,20))
plt.imshow(wordcloud, interpolation="bilinear")
plt.show()
# # plot positive tweets
# data_pos= data['text']
# wordcloud= WordCloud(max_words= 1000, width= 1600, height= 800, collocations=False).generate(data_pos)
# plt.figure(figsize=(20, 20))
plt.imshow(wordcloud, interpolation="bilinear")
plt.show()
# splitting the data into train and test
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size= 0.05, random_state= 25456167)
# # transforming data using TF-IDF VECTORIZER
# # it transforms text into meaningful representation of numbers
# # lets fit the tf-idf vectorizer
# # ngram_range(1,2)  this function will output one word two word tokens if you set (1,1) for outputting only one word tokens
vectoriser= TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)
print('No. of feature_words:', len(vectoriser.get_feature_names()))
# # transform the data using tf-idf vectorizer
X_train= vectoriser.transform(X_train)
X_test= vectoriser.transform(X_test)
# # After training the model lets evaluate to check how the model is performing
# we shall use Accuracy score, confusion matrix ROC AUC curve
def model_Evaluate(model):
    y_pred= model.predict(X_test)
    print(classification_report(y_test, y_pred))
# # compute and plot confusion matri
    cf_matrix= confusion_matrix(y_test, y_pred)
    categories= ['Negative', 'Positive']    
    group_names= ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages= ['{0:.2}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot= labels, cmap= 'Blues',fmt= '', xticklabels= categories, yticklabels= categories)
    plt.xlabel("Predicted values", fontdict= {'size':14}, labelpad= 10)
    plt.ylabel("Actual values", fontdict= {'size':14}, labelpad= 10)
    plt.title("Confusion Matrix", fontdict= {'size':18}, pad= 20)

# ## MODELLING
# # modelling using Bernoulli Naive Bayes
BNBmodel= BernoulliNB()
BNBmodel.fit(X_train, y_train)
model_Evaluate(BNBmodel)
y_pred1= BNBmodel.predict(X_test)


# # print the roc curve
# fpr, tpr, thresholds= roc_curve(y_test, y_pred1)
# roc_auc= auc(fpr, tpr)
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC CURVE')
# plt.legend(loc="lower right")
# plt.show()

# #Model2 SVM
SVCmodel= LinearSVC()
SVCmodel.fit(X_train, y_train)
model_Evaluate(SVCmodel)
y_pred2= SVCmodel.predict(X_test)
# plot the roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred2)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.show()

# # Model 3 logistic Regression
# LRmodel= LogisticRegression(c= 2, max_iter= 1000, n_jobs=_1)
# LRmodel.fit(X_train, y_train)
# model_Evaluate(LRmodel)
# y_pred3= LRmodel.predict(X_test)
# # roc curve
# fpr, tpr, thresholds = roc_curve(y_test, y_pred3)
# roc_auc = auc(fpr, tpr)
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC CURVE')def model_Evaluate(model):
#     y_pred= model.predict(X_test)
# print(classification_report(y_test, y_pred))
# # compute and plot confusion matrix
# cf_matrix= confusion_matrix(y_test, y_pred)
# categories= ['Negative', 'Positive']
# group_names= ['True Neg', 'False Pos', 'False Neg', 'True Pos']
# group_percentages= ['{0:.2}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
# labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]
# labels = np.asarray(labels).reshape(2,2)
# sns.heatmap(cf_matrix, annot= labels, cmap= 'Blues',fmt= '', xticklabels= categories, yticklabels= categories)
# plt.xlabel("Predicted values", fontdict= {'size':14}, labelpad= 10)
# plt.ylabel("Actual values", fontdict= {'size':14}, labelpad= 10)
# plt.title("Confusion Matrix", fontdict= {'size':18}, pad= 20)

# ## MODELLING
# # modelling using Bernoulli Naive Bayes
# BNBmodel= BernoulliNB()
# BNBmodel.fit(X_train, y_train)
# model_Evaluate(BNBmodel)
# y_pred1= BNBmodel.predict(X_test)
# # print(y_pred1)

# # print the roc curve
# fpr, tpr, thresholds= roc_curve(y_test, y_pred1)
# roc_auc= auc(fpr, tpr)
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC CURVE')
# plt.legend(loc="lower right")
# plt.show()
# #Model2 SVM
# SVCmodel= LinearSVC()
# SVCmodel.fit(X_train, y_train)
# model_Evaluate(SVCmodel)
# y_pred2= SVCmodel.predict(X_test)
# # plot the roc curve
# fpr, tpr, thresholds = roc_curve(y_test, y_pred2)
# roc_auc = auc(fpr, tpr)
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC CURVE')
# plt.legend(loc="lower right")
# plt.show()

# # Model 3 logistic Regression
# LRmodel= LogisticRegression(c= 2, max_iter= 1000, n_jobs=_1)
# LRmodel.fit(X_train, y_train)
# model_Evaluate(LRmodel)
# y_pred3= LRmodel.predict(X_test)
# # roc curve
# fpr, tpr, thresholds = roc_curve(y_test, y_pred3)
# roc_auc = auc(fpr, tpr)
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC CURVE')
# plt.legend(loc="lower right")
# plt.show()




