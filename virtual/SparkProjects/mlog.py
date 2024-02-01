### using MLlib logistic regression in apache spark
import findspark
findspark.init()
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from splearn.preprocessing import SparkLabelEncoder
import matplotlib.pyplot as plt
import numpy as np
spark= SparkSession.builder.appName('ml-bank').getOrCreate()
# load the data
df=spark.read.csv('/home/fridah/Downloads/bank.csv', header= True, inferSchema= True)
df.head(5)
df.show()
df.printSchema()
pd.DataFrame(df.take(5), columns=df.columns).transpose()
df=df.drop('day', 'month')
# print(df.head(5))
# print(df.describe())
# convert  the string values into numerical and vector assembler into features
features= ['job', 'marital', 'default','housing', 'loan', 'contact', 'poutcome']
num_features= ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
label= 'deposit'

encoder = OneHotEncoder(inputCol="index", outputCol="encoding")
StringIndexer(inputCol="job", outputCol="job_encoded")
encoder.setDropLast(False)
ohe = encoder.fit(indexer)
indexer = ohe.transform(indexer)

OneHotEncoder(dropLast=False, inputCol="jobencoded", outputCol="jobvec")

stringIndexer = StringIndexer(inputCol="job", outputCol="jobIndex")
model = stringIndexer.fit(df)
indexed = model.transform(df)
encoder = OneHotEncoder(dropLast=False, inputCol="jobIndex", outputCol="jobVec")
ohe = encoder.fit(indexer)
# indexer = ohe.transform(indexer)
encoded = encoder.transform(indexed)
encoded.show(2)


encoded = encoder.transform(indexer)
encoded.show(2)

# # pipeline stages
# for features in cat_features:
#     #index categorical features
#     string_indexer= StringIndexer(inputCol=features, outputCol=features + "_index")
#     # one hot encode categorical features
#     encoder= OneHotEncoder(inputCols=[string_indexer.getOutputCol()], outputCols=[features + "_class_vec"])
#     # append pipeline stages
#     stages += [string_indexer, encoder]
# # index label feature
# label_str_index= StringIndexer(inputCol=label, outputCol= "label_index")
# unscaled_features= select_features_to_scaled(df=df, lower_skew=-2, upper_skew=2, dtypes='int32', drop_cols=['id'])
# unscaled_assembler= VectorAssembler(inputCols=unscaled_features, outputCol="unscaled_features")
# stages += [unscaled_assembler, scaler]
# # create list of numeric features that are not being scaled
# num_scaled_diff_list= list(set(num_features) - set(unscaled_features))
# # assemble or concat the categorical features in numeric features
# assembler_inputs = [feature + "_class_vec" for feature in cat_features] + num_unscaled_diff_list
# assembler= VectorAssembler(inputCols=assembler_inputs, outputCol="assembled_inputs")
# stages += [label_str_index, assembler]
# assembler_final= VectorAssembler(inputCols=["scaled_features", "assembled_inputs"], outputCol="features")
# stages += [assembler_final]

# # Pipeline to chain multiple Transformers and Estimators 
# # together to specify our machine learning workflow
# pipeline= Pipeline(stages=stages)
# pipeline_model= pipeline.fit(df)
# df_transform= pipeline_model.transform(df)
# df_transform.limit(5).toPandas()
# df.printSchema
# pipeline.fit_transform(df)
# selectedCols= ['label', 'features'] + cols
# df= df.select(selectedCols)
# df.printSchema

# train, test = df.randomSplit([0.7, 0.3])
# print("Training Dataset Count: " + str(train.count()))
# print("Test Dataset Count: " + str(test.count()))
# # train the model
# lr=LogisticRegression(featuresCol= 'features', labelCol= 'label', maxIter=10)
# lrModel= lr.fit(train)
# # obtain coeffients
# beta= np.sort(lrModel.coeffients)
# plt.plot(beta)
# plt.ylabel('Beta Coefficients')
# plt.show()
