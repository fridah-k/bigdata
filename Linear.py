import findspark
findspark.init()
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import isnan, when, count, col
from pyspark.ml.regression import LinearRegression
import pyspark.sql.functions as F

from pyspark.sql import SparkSession
spark= SparkSession.builder.appName('loan').getOrCreate()
df= spark.read.csv('/home/fridah/Downloads/Loan_Train.csv', inferSchema= True, header= True)
# print(df.columns)
# df.show()
df.describe().show()
# checking for null values
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()
# Drop all the columns that are not needed.
# Also drp function only accumulates 4 arguments
cleanData = df.drop('Gender','Married', 'Dependents', 'Self_Employed')
cleanData.show()

# Replacing the null values with zeros in columns that have intergers
replaceData = cleanData.fillna(value=0, subset=["LoanAmount", "Loan_Amount_Term", "Credit_History"])
replaceData.show()

# If you want to drop all the columns with null values:
# def drop_null_columns(df):
#     null_counts = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]).collect()[0].asDict()
#     to_drop = [k for k, v in null_counts.items() if v > 0]
#     df= df.drop(*to_drop)
#     return df
# drop_null_columns(df).show()

# Feature transformation
featureassembler= VectorAssembler(inputCols=["ApplicantIncome","LoanAmount","Credit_History"], outputCol= "Independent Features")
output= featureassembler.transform(replaceData)
output.show()
# splitted our data into indepent features dependent
finalized_data= output.select("Independent Features", "Loan_Amount_Term")
finalized_data.show()
# split our data into train and test
train_data, test_data= finalized_data.randomSplit([0.75, 0.25])
regressor= LinearRegression(featuresCol= "Independent Features", labelCol= "Loan_Amount_Term")
# Trained our data
regressor= regressor.fit(train_data)
regressor.coefficients
regressor.intercept
pred_results= regressor.evaluate(test_data)
pred_results.predictions.show()









