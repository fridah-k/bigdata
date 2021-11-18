# Predicting Flight Delays.
import findspark
findspark.init()
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, Imputer
import pyspark.ml as ml
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml import PipelineModel
from pyspark.mllib.linalg import Vectors
from pyspark.sql.functions import isnan, when, count, col
from pyspark.ml.feature import ChiSqSelector

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from pyspark.ml import Pipeline

from pyspark.sql import SparkSession
spark= SparkSession.builder.appName('flights').getOrCreate()
# load the data
df=spark.read.csv("/home/fridah/Downloads/DelayedFlights.csv", inferSchema= True, header= True)
df.show(5)
##  EXPLORE THE DATA

# df.printSchema()
print(df.columns)
# df.count()
# df.describe().show()
#check for null values
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()
# fill the missing values with zeros and one
df.na.fill(value=0,subset=["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]).show()
#removing null values 
df=df.dropna(how = 'any', thresh = None, subset = None)
df.show()
#remove unwanted columns
df= df.drop(*('_c0', 'Year', 'Month','DayofMonth', 'DayofWeek', 'Origin', 'Dest', 'CRSDepTime',))

df.show()
# df(['UniqueCarrier'].value_counts())
# Visualisation of the data
flights['DepDate'] = pd.to_datetime(flights.Year*10000+flights.Month*100+flights.DayofMonth,format='%Y%m%d')
f,ax=plt.subplots(1,2,figsize=(20,8))
flights['Status'].value_counts().plot.pie(explode=[0.05,0.05,0.05,0,0],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Status')
ax[0].set_ylabel('')
sns.countplot('Status',order = flights['Status'].value_counts().index, data=flights,ax=ax[1])
ax[1].set_title('Status')
plt.show()

print('Status represents wether the flight was on time (0), slightly delayed (1), highly delayed (2), diverted (3), or cancelled (4)')

# indexing one column from string to numeric using string indexer
# changing Unique carrier to numeric from string. It will generate a new column name for the transformed column.
# indexer= StringIndexer(inputCol='UniqueCarrier',outputCol='IndexUniqueCarrier').fit(df)
# df_ind= indexer.transform(df)
# df_ind.show()
# #lets drop the column that has been changed that is UniqueCarrier,  and TailNum
# df= df.drop('UniqueCarrier', 'TailNum', 'CancellationCode')
# df.show()
# #changing multiple columns to numeric using string indexer
# categoricalColumns= [item[0] for item in df.dtypes if item[1].startswith('string')]
# #define a list of stages in your pipeline. The string indexer will be one stage
# stages = []
#iterate through all categorical values
for categoricalCol in categoricalColumns:
    #create a string indexer for those categorical values and assign a new name including the word 'Index'
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    #append the string Indexer to our list of stages
    stages += [stringIndexer]
#Create the pipeline. Assign the stages list to the pipeline key word stages
pipeline= Pipeline(stages=stages)
# fit the pipeline to our dataframe
pipelineModel= pipeline.fit(df)
#  tranform the dataframe
df= pipelineModel.transform(df)
df.show()

assembler= VectorAssembler(inputCols=['DepTime', 'ArrTime', 'CRSArrTime', 'FlightNum', 'ActualElapsedTime'
'CRSElapsedTime', 'AirTime', 'Distance', 'TaxiIn', 'TaxiOut', 'Cancelled', 'Diverted'], outputCol= 'features')
pipeline= Pipeline(stages= vectorAssembler,)

train, test = df.randomSplit([0.7, 0.3])
selector=ChiSqSelector(percentile=0.9, featuresCol="features", outputCol='selectedFeatures', labelCol= "label")
model=selector.fit(train)
result = model.transform(train)
train =result.select('label','selectedFeatures').withColumnRenamed('selectedFeatures', 'features')
new_test=model.transform(test)
test=new_test.select('label','selectedFeatures').withColumnRenamed('selectedFeatures', 'features')


