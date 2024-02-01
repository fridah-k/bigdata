import findspark
findspark.init()
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
spark= SparkSession.builder.appName('Bank Cleaned').getOrCreate()
df= spark.read.csv("/home/fridah/Downloads/bank_cleaned.csv")
# df.show()
df.printSchema()
df.createOrReplaceTempView("bank_cleaned.csv")
sqlDF = spark.sql("SELECT * FROM bank_cleaned.csv")
sqlDF.show()
# df.select("age", "education", "marital").show()

# select the age of pple older than 30
# df.filter['age']>30.show()
