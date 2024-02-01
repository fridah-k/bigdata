# We are using the credit card csv to find out the transactions that are Fraud and not Fraud.
# this is a classification problem.
import findspark
findspark.init()
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col

spark= SparkSession.builder.appName('creditcard').getOrCreate()
df= spark.read.csv("/home/fridah/Downloads/creditcard.csv", inferSchema= True, header= True)
df.show()
df.printSchema()
# length of the data
len(df.columns)
# To have a good look at the data 
df.select(["V"+str(x) for x in range (1, 5)]).show(10)
df.columns[1:-1]
# Transform the columns into float
for column_name in df.columns[1:-1]+["Class"]:
    df= df.withColumn(column_name,col(column_name).cast('float'))
# Renaming the columns
df= df.withColumnRenamed("Class", "label")
print(df.columns)
# we use the vector assembler to obtain our features
VectorAssembler= VectorAssembler(inputCols= df.columns[1:-1], outputCol= 'features')
df_tr= VectorAssembler.transform(df)
df_tr= df_tr.select(['features','label'])
df_tr.show(3)
# training our data on Logistic Regression
lr= LogisticRegression(maxIter=10, featuresCol="features",labelCol="label")
model= lr.fit(df_tr)
print(model.summary.areaUnderROC)
paramGrid= ParamGridBuilder().addGrid(lr.regParam, [0.1,0.01]).addGrid(lr.fitIntercept, [False,True]).addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]).build()
crossval= CrossValidator(estimator=lr,estimatorParamMaps=paramGrid,evaluator=BinaryClassificationEvaluator(),numFolds=2)
cvModel= crossval.fit(df_tr)
cvModel.avgMetrics






        


