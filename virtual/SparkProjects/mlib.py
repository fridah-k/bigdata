import findspark
findspark.init()
from pyspark import SparkContext
from pyspark import SparkConf
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
# create an instance of the spark class
spark=SparkSession.builder.appName('housing_price_model').getOrCreate()
# create a spark dataframe to input the csv file
df=spark.read.csv('/home/fridah/Downloads/cruise_ship_info.csv', inferSchema= True, header= True)
df.show (5)
#prints structure of dataframe along with datatype
df.printSchema()
df.columns
#columns identified as features are as below:
#['Cruise_line','Age','Tonnage','passengers','length','cabins','passenger_density']
#to work on the features, spark MLlib expects every value to be in numeric form
#feature 'Cruise_line is string datatype
#using StringIndexer, string type will be typecast to numeric datatype
#import library strinindexer for typecasting
indexer=StringIndexer(inputCol='Cruise_line',outputCol='cruise_cat')
indexed=indexer.fit(df).transform(df)
#above code will convert string to numeric feature and create a new dataframe
#new dataframe contains a new feature 'cruise_cat' and can be used further
#feature cruise_cat is now vectorized and can be used to fed to model
for item in indexed.head(5):
    print(item)
    print('\n')
#creating vectors from features
#Apache MLlib takes input if vector form
assembler=VectorAssembler(inputCols=['Age',
'Tonnage',
'passengers',
'length',
'cabins',
'passenger_density',
'cruise_cat'],outputCol='features')
output=assembler.transform(indexed)
output.select('features', 'crew').show(5)
#final data consist of features and label which is crew.
final_data=output.select('features','crew')
#splitting data into train and test
train_data,test_data=final_data.randomSplit([0.7,0.3])
train_data.describe().show()
test_data.describe().show()
#creating an object of class LinearRegression
#object takes features and label as input arguments
ship_lr= LinearRegression(featuresCol='features',labelCol='crew')
#pass train_data to train model
trained_ship_model=ship_lr.fit(train_data)
# evaluate the model trained for Rsquarred error
ship_results=trained_ship_model.evaluate(train_data)
print('RsquaredError:',ship_results.r2)
#R2 value shows accuracy of model is 92%
#model accuracy is very good and can be use for predictive analysis

#testing Model on unlabeled data
#create unlabeled data from test_data
#testing model on unlabeled data
unlabeled_data=test_data.select('features')
unlabeled_data.show(5)

predictions=trained_ship_model.transform(unlabeled_data)
predictions.show()
#below are the results of output from test data

